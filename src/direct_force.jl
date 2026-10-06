# O(N^2) pairwise acceleration for one target particle.
function net_acc_direct(pos::Matrix{Float64}, p_idx::Int,
                        masses::Vector{Float64}, charges::Vector{Float64},
                        kernel::AbstractKernel)::NTuple{3, Float64}
    acc_x = acc_y = acc_z = 0.0
    px = pos[1, p_idx]
    py = pos[2, p_idx]
    pz = pos[3, p_idx]

    if !(kernel isa Coulomb)
        target = ParticleProperties(masses[p_idx], 0.0)
        for j in axes(pos, 2)
            if j != p_idx
                dx = pos[1, j] - px
                dy = pos[2, j] - py
                dz = pos[3, j] - pz
                source = ParticleProperties(masses[j], 0.0)
                ax, ay, az = kernel_acceleration(kernel, target, source, dx, dy, dz)
                acc_x += ax
                acc_y += ay
                acc_z += az
            end
        end
        return (acc_x, acc_y, acc_z)
    end

    target = ParticleProperties(masses[p_idx], charges[p_idx])
    for j in axes(pos, 2)
        if j != p_idx
            dx = pos[1, j] - px
            dy = pos[2, j] - py
            dz = pos[3, j] - pz
            source = ParticleProperties(masses[j], charges[j])
            ax, ay, az = kernel_acceleration(kernel, target, source, dx, dy, dz)
            acc_x += ax
            acc_y += ay
            acc_z += az
        end
    end

    return (acc_x, acc_y, acc_z)
end
