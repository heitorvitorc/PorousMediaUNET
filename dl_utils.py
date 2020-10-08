import glob
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import gdal
import skimage

from dolfin import *

############################## FUNCTIONS ###############################

# Load images functions
def read_tiff(tiff_file):
    '''
    Read .tiff image as arrays
    '''
    data = gdal.Open(tiff_file).ReadAsArray()
    return data


def load_data(path):
    '''
    Load data 
    '''
    img_paths = sorted(glob.glob(path + '*.tif'))
    image = [np.expand_dims(read_tiff(img).astype('float32'), -1) for img in img_paths]
    image = np.concatenate(image, axis=-1)
    print("Image shape: ", image.shape, " Min value: ", image.min(), " Max value: ", image.max())
    return image

# Split image into training and testing 
def create_idx_image(rows, cols):
    '''
    Cria uma mascára - imagem de índices. Com o índice da imagem é possível navegar o patch de dados para 
    acessar uma imagem especifica desse índice, facilitando a alocação de memoria RAM
    '''
    im_idx = np.arange(rows * cols).reshape(rows, cols)
    return im_idx


def extract_patches(im_idx, patch_size, overlap):
    '''overlap range: 0 - 1 '''
    row_steps, cols_steps = int((1-overlap) * patch_size[0]), int((1-overlap) * patch_size[1])
    patches = skimage.util.view_as_windows(im_idx, patch_size, step=(row_steps, cols_steps))
    return patches



def mask_tiles(patches):
    num_blocks_r, num_blocks_c, rows_block, cols_block = patches.shape
    img = np.zeros((num_blocks_r*rows_block, num_blocks_c*cols_block), dtype=patches.dtype)
    for i in range(num_blocks_r):
        for j in range(num_blocks_c):
            img[rows_block*i:(rows_block*i+rows_block), cols_block*j:(cols_block*j+cols_block)] = patches[i,j]
    return img


def define_trn_val_tst_mask(tiles_grid_idx, grid_size=(10,10), val_tiles=None, tst_tiles=None, plot=True):
    '''
    Função que divide a imagem em patches de treinamento, validação e teste
    '''
    num_tiles_rows = grid_size[0] 
    num_tiles_cols = grid_size[1]

    tiles_idx = np.arange(len(tiles_grid_idx))
    if val_tiles and tst_tiles:
        val_tiles_idx = val_tiles
        tst_tiles_idx = tst_tiles
        trn_tiles_idx = set(tiles_idx) - set(val_tiles_idx) - set(tst_tiles_idx)
    else:        
        tiles_idx = np.random.permutation(tiles_idx)
        trn_tiles_idx = tiles_idx[:int(0.9*len(tiles_idx))]
        val_tiles_idx = tiles_idx[int(0.9*len(tiles_idx)):int(0.95*len(tiles_idx))]
        tst_tiles_idx = tiles_idx[int(0.95*len(tiles_idx)):]

    print(val_tiles_idx, tst_tiles_idx)

    tiles_numbers = np.zeros_like(tiles_grid_idx, dtype='uint8')
    for i in range(len(tiles_grid_idx)):
        tiles_numbers[i] = i

    mask = np.zeros_like(tiles_grid_idx, dtype='uint8')
    for idx in val_tiles_idx:
        mask[tiles_numbers==idx] = 1
    for idx in tst_tiles_idx:
        mask[tiles_numbers==idx] = 2

    mask = mask_tiles(mask.reshape(num_tiles_rows, num_tiles_cols, rows//num_tiles_rows, cols//num_tiles_cols))
    if plot:
        plt.figure(figsize=(5,5))
        plt.imshow(mask, cmap='PuBuGn')
        plt.axis('off')
        plt.show()
        plt.close()
    
    return mask


def save_data(mesh, t, u, p, path): #, pnlevels=10,resultspath='',tag='',cbarU=0,cbarP=0,cbarDirection = 0):
    
    # Mesh Vertices' Coordinates
    x = mesh.coordinates()[:,0]
    y = mesh.coordinates()[:,1]
    nVertices = len(x)
    
    mycmap = cm.get_cmap('jet')
    
    shape = (nVertices, 2)
    # Get Pressure and Velocity Values    
    uValues = u.compute_vertex_values(mesh)
    PressureData = p.compute_vertex_values(mesh)
    VelocityData = np.zeros(shape)
    
    # Colect velocity data in Arrays
    for j in range(0,nVertices):
        VelocityData[j,0] = uValues[j]
        VelocityData[j,1] = uValues[j+nVertices]
    
    np.save(path+'/VelocityData', VelocityData)
    np.save(path+'/PressureData', PressureData)
    

    # # Plot Pressure
    # plt.figure(num=fig, figsize=(10, 10), dpi=100, facecolor='w', edgecolor='k')
    # plt.clf()
    # pax = plot(p,title=dicTitle[1], cmap = mycmap)
    # # plt.axis('equal')
    # minP = pValues.min()
    # meanP = pValues.mean()
    # maxP = pValues.max()
    # pticks = [minP]
    # pticklabels = ['{:.0f} Pa'.format(minP)]
    # for l in range(1,pnlevels+1):
    #     levelPvalue = (l*(maxP-minP)/pnlevels) + minP
    #     pticks.append(levelPvalue)
    #     pticklabels.append('{:.0f} Pa'.format(levelPvalue))
    
    # if cbarDirection == 1:
    #     cbarP = plt.colorbar(pax,orientation='vertical',cmap = mycmap) #
    #     cbarP.set_ticks(pticks)
    #     cbarP.ax.set_yticklabels(pticklabels)
    # else:
    #     cbarP = plt.colorbar(pax,orientation='horizontal',cmap = mycmap) #
    #     cbarP.set_ticks(pticks)
    #     cbarP.ax.set_xticklabels(pticklabels)   
    
    # # Save Figure as .PNG file
    # if resultspath != '':
    #     plt.savefig(resultspath+tag+'_Pressure_t='+'{:.2f}'.format(t) +'.png')
    
    # # Plot Velocities
    # plt.figure(num=fig+1, figsize=(10, 10), dpi=100, facecolor='w', edgecolor='k')
    # plt.clf()
    # uax = plot(u,title=dicTitle[2], cmap = mycmap)
    # dpdx = ((pValues[len(pValues)-1]-pValues[1])/x.max())
    
    # # Calculate Arrow Sizes
    # C = np.hypot(uXYValues[:,0], uXYValues[:,1])
    # # plt.axis('equal')
    # minVel = '{:.0f} m/s'.format(C.min()) 
    # meanVel = '{:.3f} m/s'.format(C.mean())
    # maxVel = '{:.3f} m/s'.format(C.max())
    
    # if cbarDirection == 1:
    #     cbarU = plt.colorbar(uax,orientation='vertical', cmap = mycmap) #,
    #     cbarU.set_ticks([C.min(), C.mean(), C.max()])    
    #     cbarU.ax.set_yticklabels([minVel, meanVel, maxVel])
    # else:
    #     cbarU = plt.colorbar(uax,orientation='horizontal', cmap = mycmap) #,
    #     cbarU.set_ticks([C.min(), C.mean(), C.max()])    
    #     cbarU.ax.set_xticklabels([minVel, meanVel, maxVel])
    
    # # Save Figure as .PNG file
    # if resultspath != '':
    #     plt.savefig(resultspath+tag+'_Velocities_t='+'{:.2f}'.format(t) +'.png')
    
    # return cbarU, cbarP, uXYValues, pValues, nVertices;
