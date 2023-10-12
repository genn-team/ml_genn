from .communicator import Communicator

try:
    from mpi4py import MPI
except ImportError:
    _has_mpi4py = False
else:
    _has_mpi4py = True
    
class MPI(Communicator):
    """Implementation of Communicator which uses mpi4py 
    for parallel communications between ranks
    """
    def __init__(self):
        if not _has_mpi4py:
            raise ImportError("mpi4py is required to use MPI communicator")
        
        # Get communicator
        self.comm = MPI.COMM_WORLD

        # Get our rank and number of ranks
        self._rank = self.comm.Get_rank()
        self._num_ranks = self.comm.Get_size()
        
    def barrier(self):
        """Wait for all ranks to reach this point in execution before continuing
        """
        self.comm.Barrier()
    
    def broadcast(self, data, root: int):
        """Broadcast data from root to all ranks
        Args:
            data: Data to broadcast
            root: Index of node to broadcast from
        """
        self.comm.Bcast(data, root)

    def reduce_sum(self, value):
        """Calculates the sum of value across all ranks
        Args:
            value: Value to sum up
        Returns:
            Sum of value across all ranks
        """
        return self.comm.allreduce(sendobj=value, op=MPI.SUM)

    def rank(self):
        """Gets index of this rank

        Returns:
            int: rank
        """
        return self._rank

    def num_ranks(self):
        """Gets total number of ranks

        Returns:
            int: num_ranks
        """
        return self._rank