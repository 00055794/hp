"""Quick test of region grid lookup logic"""
import json

# Test coordinates
lat, lon = 51.1694, 71.4704
grid_size = 0.01

# Calculate grid cell (matching notebook logic)
lat_grid = int(round(lat / grid_size))
lon_grid = int(round(lon / grid_size))
grid_key = f"{lat_grid},{lon_grid}"

print(f"Test coordinates: ({lat}, {lon})")
print(f"Grid cell: {grid_key}")

# Load region grid lookup
with open('region_grid_lookup.json', 'r', encoding='utf-8') as f:
    region_grid_data = json.load(f)

# Extract the grid dictionary
region_grid_lookup = region_grid_data.get('grid', {})
print(f"Total grid cells loaded: {len(region_grid_lookup)}")

# Look up region name from grid
region_name = region_grid_lookup.get(grid_key)
print(f"Region name from grid: {region_name}")

# Load region encoder
with open('region_grid_encoder.json', 'r', encoding='utf-8') as f:
    region_encoder = json.load(f)

# Encode region name to integer
if region_name:
    region_code = region_encoder.get(region_name, -1)
    print(f"Region code from encoder: {region_code}")
else:
    print("Region not found in grid!")

print("\n✅ If region_code = 35, the lookup works correctly!")
