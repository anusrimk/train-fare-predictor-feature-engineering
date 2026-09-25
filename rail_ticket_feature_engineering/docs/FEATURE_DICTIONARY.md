# Feature Dictionary

## Raw fields
- `insert_date`: timestamp when the listing was inserted.
- `origin`: origin city/station label.
- `destination`: destination city/station label.
- `start_date`: departure timestamp.
- `end_date`: arrival timestamp.
- `train_type`: train/service category.
- `price`: regression target.
- `train_class`: passenger class.
- `fare`: fare/product category.

## Engineered fields
- `departure_hour`
- `arrival_hour`
- `departure_weekday`
- `arrival_weekday`
- `departure_month`
- `departure_dayofyear`
- `is_weekend`
- `departure_period`
- `arrival_period`
- `season`
- `journey_duration_hours`
- `advance_booking_hours`
- `is_overnight`
- `route_key`
- `route_frequency`
- `origin_frequency`
- `destination_frequency`
- `route_length_proxy`
- `same_day_journey`
- `train_class_x_train_type`
- `route_x_train_type`
