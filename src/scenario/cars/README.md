# Car Scenario

*Data Modalities: text, images, audio, and tables.*

## Overview

This scenario simulates a comprehensive car diagnostics dataset that contains information about vehicles, including their specifications, reported issues, and diagnostic data. The diagnostic component includes three possible modalities per car: audio recordings of engine/brake sounds (audio), images showing vehicle damage or condition (images), and text complaints describing issues (text). Each car may have data in up to three of these modalities.

In some cases, multiple modalities reflect the same underlying problem (e.g., brake issues detected in both audio recordings and complaint text), whereas in others, they may indicate distinct issues or co-occurring conditions. This scenario explores questions concerning the presence and co-occurrence of vehicle problems that are identifiable through multimodal data—specifically images, audio, and text—using language models for analysis and inference.

## Schema

The schema has four tables:
```
Car(car_id, year, mileage, fuel_type, transmission, vin, registration_date, country, number_plate, previous_owners)
CarImage(car_id, image_id, image_path, damage_status)
CarAudio(car_id, audio_id, audio_path, generic_problem, detailed_problem)
CarComplaint(car_id, complaint_id, summary, component_class, crash, fire, numberOfInjuries)
```

The ground truth and corresponding labels are provided in the denormalized file `files/cars/data/full_data/car_data_denormalized.csv`.
This file is not designed for multimodal data processing; instead, it serves to simplify ground truth generation through conventional SQL queries.

The `Car` table stores vehicle specifications and background information, including `year` (manufacturing year), `mileage` (odometer reading), `fuel_type` (Gasoline/Diesel/Hybrid/Electric/Plug-in Hybrid), `transmission` (Automatic/Manual/CVT), `vin` (Vehicle Identification Number), `registration_date`, `country` (country of registration), `number_plate` (license plate), and `previous_owners` (number of previous owners). Cars have 0, 1, 2, or 3 modalities assigned, with each modality potentially indicating different issues or a healthy state.

The `CarImage` table contains paths to vehicle images (`image_path`), each image has an `image_id` and is assigned to a car with `car_id`. The `damage_status` field indicates the type of damage visible (e.g., "no_damage", "dented", "paint_scratches", "broken_glass", or combinations separated by semicolons). Each image is assigned to only one car.

The `CarAudio` table contains paths to audio files (`audio_path`) that record engine sounds, brake sounds, or other vehicle diagnostics. Each audio has an `audio_id` and belongs to a car (`car_id`). The `generic_problem` field indicates the problem category (e.g., "startup state", "idle state", "braking state"), and `detailed_problem` provides more specific information (e.g., "normal_engine_startup", "bad_ignition", "worn_out_brakes"). Each audio file is assigned to only one car.

The `CarComplaint` table contains text complaints (`summary`) describing vehicle issues reported by owners. Each complaint has a `complaint_id` and belongs to a car (`car_id`). The `component_class` field categorizes the issue (e.g., "ENGINE", "SERVICE BRAKES", "STEERING", "ELECTRICAL SYSTEM"). Additional fields include `crash` (boolean indicating if a crash was involved), `fire` (boolean indicating if fire was involved), and `numberOfInjuries` (count of injuries). Each complaint is assigned to only one car.

## Queries

- Which cars have brake-related complaints in our database?

- Find cars with available audio recordings that show normal engine startup and have no visible damage in their images.

- Find five cars registered in the USA that have engine problems according to their audio recordings (non-normal audio).

- What is the average mileage of cars with steering-related complaints?

- Count cars with automatic transmission whose audio and images suggest brake or engine issues.

- Find cars that have issues according to at least one modality but are healthy according to another modality, consider only cars with two or more available modalities. Consider only audio recordings, images, and text complaints.

- Find cars with any reported issues in our database (consider all audio recordings, images, and text complaints)?

- Find hundred cars with crash-related complaints and more than 2 previous owners.

- Find cars that have issues according to both audio recordings and images (semantic filter/join with audio and image in one prompt).

- For all complaints, generate a component category from a given list of component classes.

## Ground Truth

- `SELECT year, mileage, fuel_type, transmission, country, car_id FROM car WHERE component_class LIKE '%BRAKE%' OR summary LIKE '%brake%';`

- `SELECT year, mileage, fuel_type, transmission, country, car_id FROM car WHERE generic_problem = 'startup state' AND detailed_problem = 'normal_engine_startup' AND damage_status = 'no_damage';`

- `SELECT year, mileage, fuel_type, transmission, country, car_id FROM car WHERE country = 'USA' AND generic_problem = 'idle state' AND detailed_problem NOT IN ('normal_engine_idle') LIMIT 5;`

- `SELECT AVG(mileage) AS average_mileage FROM car WHERE component_class LIKE '%STEERING%' OR summary LIKE '%steering%';`

- `SELECT transmission, COUNT(*) AS count FROM car WHERE transmission = 'Automatic' AND (detailed_problem NOT IN ('normal_engine_startup', 'normal_engine_idle', 'normal_brakes') OR damage_status != 'no_damage' OR component_class NOT IN ('UNKNOWN')) GROUP BY transmission;`

- `SELECT year, mileage, fuel_type, transmission, country, car_id FROM car WHERE (detailed_problem NOT IN ('normal_engine_startup', 'normal_engine_idle', 'normal_brakes') OR damage_status != 'no_damage' OR component_class NOT IN ('UNKNOWN')) AND (detailed_problem IN ('normal_engine_startup', 'normal_engine_idle', 'normal_brakes') OR damage_status = 'no_damage' OR component_class = 'UNKNOWN') ORDER BY year ASC LIMIT 1;`

- `SELECT year, mileage, fuel_type, transmission, country, car_id FROM car WHERE detailed_problem NOT IN ('normal_engine_startup', 'normal_engine_idle', 'normal_brakes') OR damage_status != 'no_damage' OR component_class NOT IN ('UNKNOWN');`
