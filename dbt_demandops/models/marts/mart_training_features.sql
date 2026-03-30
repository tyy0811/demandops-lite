-- Training-ready feature base with temporal columns.
-- Lag features (1h, 24h, 168h) and rolling mean are computed downstream
-- in Polars -- they require ordered window functions over zone partitions
-- that are more naturally expressed via shift() on sorted groups.
--
-- day_of_week convention: 0=Mon, 6=Sun (Python datetime.weekday()).
-- DuckDB extract(dow) returns 0=Sun, so we convert: (dow + 6) % 7.
-- See DECISIONS.md #4.
select
    zone_id,
    hour_ts,
    trip_count,
    avg_fare,
    avg_distance,
    extract(hour from hour_ts) as hour_of_day,
    ((extract(dow from hour_ts) + 6) % 7) as day_of_week,
    case when ((extract(dow from hour_ts) + 6) % 7) >= 5
         then 1 else 0
    end as is_weekend,
    extract(month from hour_ts) as month
from {{ ref('int_zone_hour_dense') }}
