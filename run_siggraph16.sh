#PBS -l nodes=dell01:ppn=10
#PBS -q workq
#PBS -N siggraph16
cd   $PBS_O_WORKDIR
echo $PBS_NODEFILE

cd /largedisk/home/plwang/siggraph2016_colorization
for i in {1..500}
do
  lua colorize.lua "./img/val_256_gray/Places365_val_"$i".jpg" "./img_out/Places365_val_"$i".jpg" "colornet_imagenet.t7"
done
