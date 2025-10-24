"""
Region Grid Lookup - Load and use region grid for predictions
"""
import json
import os
import numpy as np

class RegionGridLookup:
    """Load and use the region grid lookup for assigning regions based on coordinates"""
    
    def __init__(self, grid_file='region_grid_lookup.json'):
        """
        Initialize the region grid lookup
        
        Parameters:
        -----------
        grid_file : str
            Path to the region_grid_lookup.json file
        """
        self.grid_file = grid_file
        self.grid = {}
        self.grid_size = 0.01
        self.bounds = {}
        self.region_counts = {}
        self.region_to_code = {}
        
        if os.path.exists(grid_file):
            self.load_grid(grid_file)
    
    def load_grid(self, grid_file):
        """Load grid from JSON file"""
        with open(grid_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert string keys back to tuples
        self.grid = {tuple(map(int, k.split(','))): v for k, v in data['grid'].items()}
        self.grid_size = data['grid_size']
        self.bounds = data['bounds']
        self.region_counts = data['region_counts']
        
        # Load the ACTUAL encoder mapping used during training (CRITICAL!)
        # Try multiple locations for the encoder file
        grid_dir = os.path.dirname(os.path.abspath(grid_file))
        encoder_paths = [
            os.path.join(grid_dir, 'region_grid_encoder.json'),
            os.path.join(os.path.dirname(__file__), 'region_grid_encoder.json'),
            'region_grid_encoder.json'
        ]
        
        encoder_file = None
        for path in encoder_paths:
            if os.path.exists(path):
                encoder_file = path
                break
        
        if encoder_file:
            with open(encoder_file, 'r', encoding='utf-8') as f:
                self.region_to_code = json.load(f)
            print(f"✓ Loaded region grid: {len(self.grid)} cells, {len(self.region_to_code)} regions")
            print(f"✓ Loaded encoder mapping from: {encoder_file}")
        else:
            # Fallback: create mapping (but this will NOT match training!)
            unique_regions = sorted(set(self.grid.values()))
            self.region_to_code = {region: idx for idx, region in enumerate(unique_regions)}
            print(f"⚠️  WARNING: Using fallback encoding - may not match training!")
            print(f"   Encoder file not found in: {encoder_paths}")
            print(f"   Please ensure region_grid_encoder.json exists in app/ folder")
            print(f"✓ Loaded region grid: {len(self.grid)} cells, {len(unique_regions)} regions")
    
    def lookup_region(self, lat, lon, fallback='Unknown'):
        """
        Look up region for given coordinates
        
        Parameters:
        -----------
        lat : float
            Latitude
        lon : float
            Longitude
        fallback : str
            Default region if not found
            
        Returns:
        --------
        str : Region name
        """
        lat_grid = round(lat / self.grid_size)
        lon_grid = round(lon / self.grid_size)
        
        return self.grid.get((lat_grid, lon_grid), fallback)
    
    def lookup_region_code(self, lat, lon, fallback_code=-1):
        """
        Look up region code (integer) for given coordinates
        
        Parameters:
        -----------
        lat : float
            Latitude
        lon : float
            Longitude
        fallback_code : int
            Default code if not found
            
        Returns:
        --------
        int : Region code
        """
        region_name = self.lookup_region(lat, lon, fallback='Unknown')
        return self.region_to_code.get(region_name, fallback_code)
    
    def batch_lookup_regions(self, lat_lon_pairs, fallback='Unknown'):
        """
        Batch lookup regions for multiple coordinate pairs
        
        Parameters:
        -----------
        lat_lon_pairs : list of tuples
            List of (latitude, longitude) tuples
        fallback : str
            Default region if not found
            
        Returns:
        --------
        list : List of region names
        """
        return [self.lookup_region(lat, lon, fallback) for lat, lon in lat_lon_pairs]
    
    def batch_lookup_region_codes(self, lat_lon_pairs, fallback_code=-1):
        """
        Batch lookup region codes for multiple coordinate pairs
        
        Parameters:
        -----------
        lat_lon_pairs : list of tuples
            List of (latitude, longitude) tuples
        fallback_code : int
            Default code if not found
            
        Returns:
        --------
        list : List of region codes (integers)
        """
        return [self.lookup_region_code(lat, lon, fallback_code) for lat, lon in lat_lon_pairs]


# Global instance (loaded once)
_region_grid_instance = None

def get_region_grid_lookup(grid_file='../region_grid_lookup.json'):
    """Get or create the global region grid lookup instance"""
    global _region_grid_instance
    
    if _region_grid_instance is None:
        # Try multiple locations
        search_paths = [
            grid_file,
            'region_grid_lookup.json',
            '../region_grid_lookup.json',
            os.path.join(os.path.dirname(__file__), '..', 'region_grid_lookup.json')
        ]
        
        for path in search_paths:
            if os.path.exists(path):
                _region_grid_instance = RegionGridLookup(path)
                break
        
        if _region_grid_instance is None:
            print("⚠️  Warning: region_grid_lookup.json not found. Region grid lookup disabled.")
            _region_grid_instance = RegionGridLookup()  # Empty instance
    
    return _region_grid_instance
