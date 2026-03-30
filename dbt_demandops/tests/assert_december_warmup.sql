-- December data must be present (required for 168h lag features).
-- Returns 1 row if December is missing. Test passes if 0 rows.
select 1
where (
    select count(*)
    from {{ ref('int_zone_hour_dense') }}
    where hour_ts >= timestamp '2023-12-01 00:00:00'
      and hour_ts < timestamp '2024-01-01 00:00:00'
) = 0
