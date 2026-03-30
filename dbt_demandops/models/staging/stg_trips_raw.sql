-- Staging: filter raw TLC Yellow Taxi trips.
-- Mirrors: demandops/data/adapters/taxi.py TaxiAdapter.prepare_hourly_history()
with source as (
    select * from {{ source('tlc', 'yellow_tripdata') }}
),

filtered as (
    select
        "PULocationID" as zone_id,
        date_trunc('hour', tpep_pickup_datetime) as hour_ts,
        total_amount,
        trip_distance
    from source
    where tpep_pickup_datetime is not null
      and tpep_dropoff_datetime is not null
      and "PULocationID" is not null
      and "PULocationID" between 1 and 263
      and total_amount is not null
      and total_amount >= 0
      and trip_distance is not null
      and trip_distance >= 0
      and tpep_pickup_datetime >= timestamp '2023-12-01 00:00:00'
      and tpep_pickup_datetime < timestamp '2024-03-01 00:00:00'
)

select * from filtered
