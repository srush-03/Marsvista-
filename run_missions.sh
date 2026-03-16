#!/bin/bash

set -e
set -o pipefail

SEED_ROOT="seeds"
MISSION_SCRIPT="pipeline/mission_controller.py"
BATTERY_SCRIPT="scripts/check_battery.py"
RETURN_SCRIPT="scripts/return_to_base.py"

BATTERY_LIMIT=20

echo "================================="
echo " AUTONOMOUS DRONE MISSION MANAGER"
echo "================================="

while true
do

for seed_folder in "$SEED_ROOT"/*; do

    [ -d "$seed_folder" ] || continue

    echo "---------------------------------"
    echo "Starting mission: $seed_folder"
    echo "---------------------------------"

    # ----------------------------------
    # Pre-flight battery check
    # ----------------------------------

    battery=$(python3 "$BATTERY_SCRIPT")

    echo "Battery level: $battery%"

    if [ "$battery" -lt "$BATTERY_LIMIT" ]; then

        echo "Battery below threshold. Returning to base."

        python3 "$RETURN_SCRIPT"

        exit 0
    fi


    # ----------------------------------
    # Launch mission
    # ----------------------------------

    python3 "$MISSION_SCRIPT" --seeds "$seed_folder" &

    MISSION_PID=$!

    echo "Mission process started: $MISSION_PID"


    # ----------------------------------
    # Monitor mission
    # ----------------------------------

    while kill -0 "$MISSION_PID" 2>/dev/null; do

        battery=$(python3 "$BATTERY_SCRIPT")

        if [ "$battery" -lt "$BATTERY_LIMIT" ]; then

            echo "Battery dropped below 20%"
            echo "Aborting mission..."

            kill "$MISSION_PID" || true

            python3 "$RETURN_SCRIPT"

            exit 0

        fi

        sleep 2

    done


    echo "Mission completed."


    # ----------------------------------
    # Completion flag check
    # ----------------------------------

    if [ -f mission_complete.flag ]; then

        echo "Mission success confirmed"

        rm mission_complete.flag

    fi


    # ----------------------------------
    # Return to base
    # ----------------------------------

    echo "Returning drone to base..."

    python3 "$RETURN_SCRIPT"


    # ----------------------------------
    # Archive mission data
    # ----------------------------------

    timestamp=$(date +"%Y%m%d_%H%M%S")

    archive_dir="data/archive/$timestamp"

    mkdir -p "$archive_dir"

    if [ -d data/detections ]; then
        mv data/detections "$archive_dir"/
    fi

    if [ -d data/logs ]; then
        mv data/logs "$archive_dir"/
    fi

    if [ -d data/keyframes ]; then
        mv data/keyframes "$archive_dir"/
    fi

    mkdir -p data/detections
    mkdir -p data/logs
    mkdir -p data/keyframes

    echo "Mission data archived to $archive_dir"

done

echo "All seed missions finished."

done