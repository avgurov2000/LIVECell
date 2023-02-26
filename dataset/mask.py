from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import pycocotools._mask as _mask

class maskUtils(object):
    
    iou         = _mask.iou
    merge       = _mask.merge
    frPyObjects = _mask.frPyObjects

    @staticmethod
    def encode(bimask):
        if len(bimask.shape) == 3:
            return _mask.encode(bimask)
        elif len(bimask.shape) == 2:
            h, w = bimask.shape
            return _mask.encode(bimask.reshape((h, w, 1), order='F'))[0]
        
    @staticmethod
    def decode(rleObjs):
        if type(rleObjs) == list:
            return _mask.decode(rleObjs)
        else:
            return _mask.decode([rleObjs])[:,:,0]
        
    @staticmethod
    def area(rleObjs):
        if type(rleObjs) == list:
            return _mask.area(rleObjs)
        else:
            return _mask.area([rleObjs])[0]
        
    @staticmethod
    def toBbox(rleObjs):
        if type(rleObjs) == list:
            return _mask.toBbox(rleObjs)
        else:
            return _mask.toBbox([rleObjs])[0]
        
        
class CocoSegmentation(object):
    
    @staticmethod
    def annToMask(shape, ann):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = CocoSegmentation.annToRLE(shape, ann)
        m = maskUtils.decode(rle)
        return m
    
    @staticmethod
    def annToRLE(shape, ann):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        h, w = shape
        segm = ann['segmentation']
        if type(segm) == list:
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, h, w)
            rle = maskUtils.merge(rles)
        elif type(segm['counts']) == list:
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, h, w)
        else:
            # rle
            rle = ann['segmentation']
        return rle
    
    @staticmethod
    def showAnns(shape, anns):
        """
        Display the specified annotations.
        :param anns (array of object): annotations to display
        :return: None
        """
        if len(anns) == 0:
            return 0
        if 'segmentation' in anns[0] or 'keypoints' in anns[0]:
            datasetType = 'instances'
        elif 'caption' in anns[0]:
            datasetType = 'captions'
        else:
            raise Exception('datasetType not supported')
        if datasetType == 'instances':
            ax = plt.gca()
            ax.set_autoscale_on(False)
            polygons = []
            color = []
            for ann in anns:
                c = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
                if 'segmentation' in ann:
                    if type(ann['segmentation']) == list:
                        # polygon
                        for seg in ann['segmentation']:
                            poly = np.array(seg).reshape((int(len(seg)/2), 2))
                            polygons.append(Polygon(poly))
                            color.append(c)
                    else:
                        # mask
                        h, w = shape
                        if type(ann['segmentation']['counts']) == list:
                            rle = maskUtils.frPyObjects([ann['segmentation']], h, w)
                        else:
                            rle = [ann['segmentation']]
                        m = maskUtils.decode(rle)
                        img = np.ones( (m.shape[0], m.shape[1], 3) )
                        if ann['iscrowd'] == 1:
                            color_mask = np.array([2.0,166.0,101.0])/255
                        if ann['iscrowd'] == 0:
                            color_mask = np.random.random((1, 3)).tolist()[0]
                        for i in range(3):
                            img[:,:,i] = color_mask[i]
                        ax.imshow(np.dstack( (img, m*0.5) ))

            p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
            ax.add_collection(p)
            p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=2)
            ax.add_collection(p)
        elif datasetType == 'captions':
            for ann in anns:
                print(ann['caption'])