-- Zone IDs must be between 1 and 263.
-- Returns rows that violate the bound. Test passes if 0 rows returned.
select zone_id
from {{ ref('stg_trips_raw') }}
where zone_id < 1 or zone_id > 263
