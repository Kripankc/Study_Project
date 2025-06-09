import pandas as pd
import rasterio


class DataLoader:
    # File paths to precipitation, station coordinates, and DEM
    PPT_FILE_PATH = r"D:\TUM_3rd_Sem\Study_Project\data\ppt_1D_gkd_dwd_tss.pkl"
    CSV_FILE_PATH = r"D:\TUM_3rd_Sem\Study_Project\data\ppt_1D_gkd_dwd_crds.csv"
    DEM_FILE_PATH = r"D:\TUM_3rd_Sem\Study_Project\data\DEM\srtm_de_mosaic_utm32N_1km_Cl_study_area.tif"

    @staticmethod
    def load_precipitation_data():
        # Load daily precipitation data from a pickle file
        try:
            ppt_data = pd.read_pickle(DataLoader.PPT_FILE_PATH)
            print("Precipitation data loaded.")
            return ppt_data
        except Exception as e:
            print(f"Error loading precipitation data: {e}")
            return None

    @staticmethod
    def load_elevation_data():
        # Load station coordinates and elevation data from CSV
        try:
            elev_data = pd.read_csv(DataLoader.CSV_FILE_PATH, delimiter=";", index_col=0)
            elev_data.rename(columns={"Z_SRTM": "Elevation"}, inplace=True)
            print("Elevation data loaded.")
            return elev_data
        except Exception as e:
            print(f"Error loading elevation data: {e}")
            return None

    @staticmethod
    def load_dem():
        # Load DEM raster and return array, transform, and CRS
        try:
            with rasterio.open(DataLoader.DEM_FILE_PATH) as src:
                dem_array = src.read(1)
                return dem_array, src.transform, src.crs
        except Exception as e:
            print(f"Error loading DEM: {e}")
            return None, None, None


if __name__ == "__main__":
    # Simple check to verify that data loads correctly
    ppt = DataLoader.load_precipitation_data()
    elev = DataLoader.load_elevation_data()
    dem, transform, crs = DataLoader.load_dem()

    if ppt is not None:
        print("Precipitation shape:", ppt.shape)
    if elev is not None:
        print("Elevation data shape:", elev.shape)
    if dem is not None:
        print("DEM shape:", dem.shape)

