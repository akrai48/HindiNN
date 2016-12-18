for dir_name in `ls|grep usr_`
do
echo "Current directory $dir_name"
cd $dir_name
for file in `ls|grep tiff`
do
u="_"
mv $file "$dir_name$u$file"
done
cd ..
done

