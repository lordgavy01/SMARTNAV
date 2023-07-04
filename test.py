import os
def format_size(size):
    if size >= 2**30:
        return f"{size / 2**30:.2f} GB"
    elif size >= 2**20:
        return f"{size / 2**20:.2f} MB"
    elif size >= 2**10:
        return f"{size / 2**10:.2f} KB"
    else:
        return f"{size} bytes"

def get_size(path):
    total = 0
    if os.path.isfile(path):
        total += os.path.getsize(path)
    else:
        for entry in os.listdir(path):
            child_path = os.path.join(path, entry)
            total += get_size(child_path)
    return total

path = "."  # Current directory
entries = os.listdir(path)
sorted_entries = sorted(entries, key=lambda entry: get_size(os.path.join(path, entry)), reverse=True)

for entry in sorted_entries:
    entry_path = os.path.join(path, entry)
    entry_size = get_size(entry_path)
    formatted_size = format_size(entry_size)
    print(f"{entry}: {formatted_size}")


