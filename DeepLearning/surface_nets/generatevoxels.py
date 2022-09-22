import stltovoxel
for i in range(0,100):
    stltovoxel.convert_file('bulbo_{}.stl'.format(i),'bulbo_{}.png'.format(i))