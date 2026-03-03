#include <stdexcept>
#include <string>
#include "utils.h"


int main(int argc, char **argv) {
  std::string input_partition_assignment_file = argv[1];
  std::string ids_file = argv[2];
  std::string output_partition_assignment_file = argv[3];

  if (!file_exists(input_partition_assignment_file)) {
    throw std::invalid_argument(input_partition_assignment_file + " doesn't exist");
  }
  if (!file_exists(ids_file)) {
    throw std::invalid_argument(ids_file + " doesn't exist");
  }  

  size_t num_pts, dim;
  uint8_t *partition_asignment;
  pipeann::load_bin<uint8_t>(input_partition_assignment_file, partition_asignment,
                    num_pts, dim);
  if (dim != 1) {
    throw std::invalid_argument("dim is not 1: " + std::to_string(dim));
  }

  uint32_t *ids;
  size_t num_ids;
  pipeann::load_bin<uint32_t>(ids_file, ids, num_ids, dim);
  if (dim != 1) {
    throw std::invalid_argument("dim is not 1: " + std::to_string(dim));
  }
  std::vector<uint8_t> output_pa(num_ids);
  for (size_t i = 0; i < num_ids; i++) {
    output_pa[i] = partition_asignment[ids[i]];
  }
  pipeann::save_bin<uint8_t>(output_partition_assignment_file, output_pa.data(),
                             num_ids, 1);
  return 0;
}
