-- Aggregate trips to zone x hour grain.
-- Mirrors: taxi.py hourly_agg table.
select
    zone_id,
    hour_ts,
    count(*)::integer as trip_count,
    avg(total_amount) as avg_fare,
    avg(trip_distance) as avg_distance
from {{ ref('stg_trips_raw') }}
group by zone_id, hour_ts
