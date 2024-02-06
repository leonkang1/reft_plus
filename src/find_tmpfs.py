def find_tmpfs_mountpoints():
    mountpoints = []
    with open('/proc/mounts', 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.split()
            if parts[2] == 'tmpfs':
                mountpoints.append(parts[1])
    return mountpoints
tmpfs_mounts = find_tmpfs_mountpoints()
print(f"Tmpfs mountpoints: {tmpfs_mounts}")