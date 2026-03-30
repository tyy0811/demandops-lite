-- total_amount and trip_distance must be non-negative.
-- Returns rows that violate. Test passes if 0 rows returned.
select total_amount, trip_distance
from {{ ref('stg_trips_raw') }}
where total_amount < 0 or trip_distance < 0
