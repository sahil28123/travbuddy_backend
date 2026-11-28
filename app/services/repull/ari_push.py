"""The calendar, out. (Implemented in the ARI push work item.)"""


async def enqueue_ari_push(property_id: str, room_type_id: str, from_date, to_date) -> None:
    raise NotImplementedError


async def enqueue_room_type(property_id: str, room_type_id: str) -> None:
    raise NotImplementedError


async def enqueue_property(property_id: str, *, days: int) -> dict:
    raise NotImplementedError
