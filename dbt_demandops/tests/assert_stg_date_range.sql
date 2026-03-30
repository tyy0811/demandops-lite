-- All trips must fall within Dec 2023 - Feb 2024.
-- Returns rows outside the range. Test passes if 0 rows returned.
select hour_ts
from {{ ref('stg_trips_raw') }}
where hour_ts < timestamp '2023-12-01 00:00:00'
   or hour_ts >= timestamp '2024-03-01 00:00:00'
