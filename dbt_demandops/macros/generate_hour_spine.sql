{% macro generate_hour_spine(start_ts, end_ts) %}
select unnest(
    generate_series(
        timestamp {{ start_ts }},
        timestamp {{ end_ts }},
        interval '1 hour'
    )
) as hour_ts
{% endmacro %}
