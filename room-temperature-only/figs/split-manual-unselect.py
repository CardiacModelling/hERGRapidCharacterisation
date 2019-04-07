from subprocess import call
import glob

IDs = ['herg25oc1']

for ID in IDs:
    unselected = []
    with open('./manual-unselected-validation-%s.txt' % ID, 'r') as f:
        for l in f:
            if not l.startswith('#'):
                unselected.append(l.split()[0])
    folder = './unselected-validation-%s'%ID
    folder_selected = './selected-validation-%s' % ID
    try:
        call(['mkdir', folder])
    except:
        print(folder + ' exists')
        pass
    call(['cp', './%s-autoLC-releak-zoom' % ID, folder_selected, '-r'])
    for cell in unselected:
        call(['mv', '%s/%s.png'%(folder_selected, cell), folder])
