"""
Repull — the channel manager behind HostOS's Airbnb, Booking.com, Vrbo and
tiket.com connectivity, and its door to other PMSs.

    client.py        the one place an API key becomes a request
    signature.py     what makes a webhook body trustworthy
    events.py        every webhook event, once
    connections.py   which Repull workspace is which hotel, and which listing
                     is which room type
    ari_push.py      the calendar, out
    reservations.py  bookings, in
    pms_import.py    a hotel arriving from another PMS

The guide for operators is docs/repull.md.
"""
