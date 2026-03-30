-- Every zone must have every hour in the range (no gaps).
-- Returns rows representing gaps. Test passes if 0 rows returned.
with expected_hours as (
    {{ generate_hour_spine("'2023-12-01 00:00:00'", "'2024-02-29 23:00:00'") }}
),
zone_list as (
    select distinct zone_id from {{ ref('int_zone_hour_dense') }}
),
expected as (
    select z.zone_id, h.hour_ts
    from zone_list z cross join expected_hours h
),
actual as (
    select zone_id, hour_ts from {{ ref('int_zone_hour_dense') }}
)
select e.zone_id, e.hour_ts
from expected e
left join actual a on e.zone_id = a.zone_id and e.hour_ts = a.hour_ts
where a.zone_id is null
