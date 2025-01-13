#!/bin/bash

# Fungsi untuk menampilkan struktur folder
display_folder_structure() {
    local dir="$1"
    local indent="$2"
    local show=("${!3}")
    local exclude=("${!4}")
    local items=($(ls -1 "$dir"))
    
    for ((i=0; i<${#items[@]}; i++)); do
        item="${items[$i]}"
        path="$dir/$item"
        
        # Cek apakah item ada di daftar exclude
        if [[ " ${exclude[@]} " =~ " ${item} " ]]; then
            continue  # Lewati item ini jika ada di daftar exclude
        fi
        
        # Cek apakah item ada di daftar show
        if [[ ${#show[@]} -gt 0 && ! " ${show[@]} " =~ " ${item} " ]]; then
            continue  # Lewati item ini jika tidak ada dalam daftar show
        fi
        
        if [[ $i -eq $((${#items[@]} - 1)) ]]; then
            # Jika item terakhir
            prefix="└── "
        else
            # Jika item bukan terakhir
            prefix="├── "
        fi

        # Tampilkan item
        printf "%*s%s%s" $((indent * 4)) "" "$prefix" "$item"
        
        # Jika item adalah folder, tambahkan tanda `/`
        if [ -d "$path" ]; then
            echo "/"
            # Panggil fungsi rekursif untuk folder
            display_folder_structure "$path" $((indent + 1)) show[@] exclude[@]
        else
            echo
        fi
    done
}

# Variabel default untuk path dan daftar show/exclude
root_directory="."
show=()
exclude=()

# Parse argument yang diberikan
for arg in "$@"; do
    case $arg in
        p=*)
            root_directory="${arg#*=}"  # Ambil path dari parameter p
            ;;
        s=*)
            show+=("${arg#*=}")  # Menambahkan folder yang ingin ditampilkan
            ;;
        ns=*)
            exclude+=("${arg#*=}")  # Menambahkan folder yang ingin dikecualikan
            ;;
    esac
done

# Tampilkan hasil
echo "$root_directory/"
display_folder_structure "$root_directory" 0 show[@] exclude[@]
