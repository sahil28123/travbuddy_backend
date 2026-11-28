-- A hotel can sell on Airbnb, Booking.com, Vrbo and tiket.com.
--
-- HostOS takes bookings from its own website, WhatsApp, Instagram and the front
-- desk. This adds the tables that let it also sell on the OTAs through Repull
-- (https://repull.dev): which Repull workspace is which hotel, which Repull
-- listing is which room type, a log of every webhook Repull sends, and the
-- columns that let a reservation remember which channel booking it is.
--
-- Everything here is inert until ENABLE_REPULL=true. See docs/repull.md.
--
-- Idempotent: safe to run twice.

-- ── which workspace is which hotel ───────────────────────────────────────────

create table if not exists public.repull_connections (
  property_id          uuid primary key references public.properties(id) on delete cascade,
  partner_id           uuid,
  workspace_id         text not null unique,
  status               text not null default 'pending'
                       check (status in ('pending','active','disconnected')),
  connected_providers  jsonb not null default '[]'::jsonb,
  created_at           timestamptz not null default now(),
  updated_at           timestamptz not null default now()
);

comment on table public.repull_connections is
  'One Repull sub-workspace per hotel. A webhook carries workspaceId, which is '
  'how an incoming booking or guest message finds its property.';

-- When this hotel last heard from a guest on an OTA thread. The same column the
-- WhatsApp and Instagram connections carry, written by the same code path.
alter table public.repull_connections add column if not exists last_inbound_at timestamptz;

-- ── which listing is which room type ─────────────────────────────────────────

create table if not exists public.repull_room_map (
  room_type_id       uuid primary key references public.room_types(id) on delete cascade,
  property_id        uuid not null references public.properties(id) on delete cascade,
  rate_plan_id       uuid references public.rate_plans(id) on delete set null,
  repull_listing_id  text not null unique,
  listing_name       text,
  unit_count         integer not null default 1 check (unit_count >= 1),
  created_at         timestamptz not null default now(),
  updated_at         timestamptz not null default now()
);

create index if not exists repull_room_map_property on public.repull_room_map (property_id);

comment on table public.repull_room_map is
  'A HostOS room type sold as one Repull listing. rate_plan_id is the plan whose '
  'price goes to the channels (an OTA listing has one nightly price). unit_count '
  'is the listing''s room count on Repull, cached because availableUnits may not '
  'exceed it.';

-- ── every webhook, once ──────────────────────────────────────────────────────

create table if not exists public.repull_webhook_events (
  event_id      text primary key,
  event_type    text not null,
  workspace_id  text,
  status        text not null default 'received'
                check (status in ('received','processed','ignored','failed','retrying')),
  error         text,
  payload       jsonb not null default '{}'::jsonb,
  received_at   timestamptz not null default now()
);

create index if not exists repull_webhook_events_failed
  on public.repull_webhook_events (received_at) where status = 'failed';

comment on table public.repull_webhook_events is
  'Every Repull webhook delivery, keyed on X-Repull-Event-Id (stable across '
  'retries). A failed row is reclaimable so Repull''s retry is handled, not skipped.';

-- ── a reservation remembers which channel booking it is ──────────────────────

alter table public.reservations add column if not exists external_source text;
alter table public.reservations add column if not exists external_id text;
alter table public.reservations add column if not exists external_ref text;
alter table public.reservations add column if not exists external_payload jsonb;

create unique index if not exists reservations_external_unique
  on public.reservations (external_source, external_id)
  where external_id is not null;

comment on column public.reservations.external_source is
  'Where a channel booking came from: airbnb, booking, vrbo, tiket, or a PMS '
  'imported through Repull. Null for bookings HostOS took itself.';
comment on column public.reservations.external_id is
  'The Repull reservation id. With external_source, the key every webhook '
  'replay and every import re-run is idempotent on.';
comment on column public.reservations.external_ref is
  'The confirmation code the guest sees on the OTA (HMXXXX on Airbnb, the '
  'Booking.com reservation number).';

-- ── OTA guest messages are a conversation channel ────────────────────────────
--
-- conversations.channel is constrained to the channels HostOS speaks. Replace
-- whichever check constraint names it with the same list plus 'ota'. Found by
-- definition rather than by name, because the base schema is not in this repo.

do $$
declare c record;
begin
  for c in
    select conname from pg_constraint
    where conrelid = 'public.conversations'::regclass
      and contype = 'c'
      and pg_get_constraintdef(oid) ilike '%channel%'
  loop
    execute format('alter table public.conversations drop constraint %I', c.conname);
  end loop;
  alter table public.conversations add constraint conversations_channel_check
    check (channel in ('whatsapp','instagram','website','phone','email','internal','ota'));
exception when duplicate_object then null;
end $$;

-- ── RLS: the partner app may read its own hotel's mapping, nobody writes ─────
--
-- Writes go through the service-role API (services/repull/*). The webhook log
-- has no policy at all — nothing outside the server should ever read it, for
-- the same reason as razorpay_webhook_events.

alter table public.repull_connections enable row level security;
alter table public.repull_room_map enable row level security;
alter table public.repull_webhook_events enable row level security;
alter table public.repull_webhook_events force row level security;
revoke all on public.repull_webhook_events from anon, authenticated;

do $$ begin
  create policy repull_connections_read on public.repull_connections
    for select using (owns_partner(partner_id));
exception when duplicate_object then null; end $$;

do $$ begin
  create policy repull_room_map_read on public.repull_room_map
    for select using (
      exists (select 1 from public.properties p
              where p.id = repull_room_map.property_id and owns_partner(p.partner_id))
    );
exception when duplicate_object then null; end $$;
