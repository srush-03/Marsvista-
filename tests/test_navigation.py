from navigation.lawnmower_planner import LawnmowerPlanner

planner = LawnmowerPlanner(
    area_width=10,
    area_height=10,
    row_spacing=1
)

path = planner.generate_path()

print("\nGenerated Waypoints:\n")

for waypoint in path:
    print(waypoint)