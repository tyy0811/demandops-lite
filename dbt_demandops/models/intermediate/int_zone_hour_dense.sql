-- Densify: cross-join all zones x all hours, left join actuals.
-- Ensures every (zone_id, hour_ts) pair has exactly one row.
-- Mirrors: taxi.py dense_grid table.
with hour_spine as (
    {{ generate_hour_spine("'2023-12-01 00:00:00'", "'2024-02-29 23:00:00'") }}
),

zone_spine as (
    select distinct zone_id
    from {{ ref('int_zone_hour_agg') }}
),

dense_grid as (
    select
        z.zone_id,
        h.hour_ts
    from zone_spine z
    cross join hour_spine h
)

select
    g.zone_id,
    g.hour_ts,
    coalesce(a.trip_count, 0)::integer as trip_count,
    a.avg_fare,
    a.avg_distance
from dense_grid g
left join {{ ref('int_zone_hour_agg') }} a
    on g.zone_id = a.zone_id
    and g.hour_ts = a.hour_ts
