SOURCED=0
(return 0 2>/dev/null) && SOURCED=1


ALL_CLOUDLAB_HOSTS=(
    "namanh@er117.utah.cloudlab.us"
    "namanh@er035.utah.cloudlab.us"
    "namanh@er130.utah.cloudlab.us"
    "namanh@er063.utah.cloudlab.us"
    "namanh@er007.utah.cloudlab.us" 
    "namanh@er020.utah.cloudlab.us"
    "namanh@er078.utah.cloudlab.us"
    "namanh@er056.utah.cloudlab.us"
    "namanh@er027.utah.cloudlab.us"
    "namanh@er070.utah.cloudlab.us"
    "namanh@er023.utah.cloudlab.us"
)

if [[ $SOURCED == 0 ]]; then
    echo "Cloudlab hosts are: "
    for CLOUDLAB_HOST in ${ALL_CLOUDLAB_HOSTS[@]}; do
	echo "$CLOUDLAB_HOST"
    done
    
fi

export ALL_CLOUDLAB_HOSTS
