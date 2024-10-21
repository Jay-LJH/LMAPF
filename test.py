from im_function_PIBT_1.build import lifelong_pibt_1
from map_generator import *

warehouse = Warehouse()
print(warehouse)
print(warehouse.station_map)
#    RHCR_warehouse(int seed,int num_of_robots, int rows,int cols,int env_id,
# vector<vector<int>> py_map,vector<vector<int>> station_map,std::string project_path);

rhcr=lifelong_pibt_1.RHCR_warehouse(42
            ,5, warehouse.height, warehouse.width,1,warehouse.matrix, \
                warehouse.station_map,".")
