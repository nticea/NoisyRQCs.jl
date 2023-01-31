"""
Some notes about MPDOs from https://arxiv.org/pdf/1804.09796.pdf 
    - Do not conserve positivity. Checking for positivity is an NP-hard problem
    - Alternative approaches: quantum trajectories and locally purified tensor networks (LPTNs)

"""

struct MPDO

end


function apply_circuit_mpdo(ψ0::MPS, T::Int)
    ## Transform ψ0 into an MPDO ##

    # Take the input state and add a dummy index (for now just dim=1) 

    for t in 1:T
        # Apply a layer of unitary evolution to the MPS 

        # Apply the noise channel 

        # Perform a canonicalization from L → R 

        # Truncate the virtual bonds

        # Truncate the inner indices 
    end

end