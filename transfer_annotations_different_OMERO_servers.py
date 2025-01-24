import os
import io
import json
import logging
from getpass import getpass
import numpy as np
# conda create -n GBM python=3.10 -c conda-forge
# conda install scanpy squidpy pandas pillow shapely omero-py -c conda-forge
import omero
import pandas as pd
from PIL import Image
from omero.gateway import BlitzGateway
from shapely import affinity,plotting
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import os
import fire
from omero.model.enums import UnitsLength
from omero.rtypes import  rstring, rint


log = logging.getLogger(__name__)

def create_roi(img, shapes):
    # create an ROI, link it to Image
    roi = omero.model.RoiI()
    # use the omero.model.ImageI that underlies the 'image' wrapper
    roi.setImage(img._obj)
    for shape in shapes:
        roi.addShape(shape)
    # Save the ROI (saves any linked shapes too)
    #return updateService.saveAndReturnObject(roi)
    return roi

def rgba_to_int(red, green, blue, alpha=255):
    """ Return the color as an Integer in RGBA encoding """
    r = red << 24
    g = green << 16
    b = blue << 8
    a = alpha
    rgba_int = r+g+b+a
    if (rgba_int > (2**31-1)):       # convert to signed 32-bit int
        rgba_int = rgba_int - 2**32
    return rgba_int

def write_one_roi(omero_username, omero_password, omero_host, polygon, image_id):
    
    #print(f"Connecting to {omero_srv} in port {omero_port}")
    #print(f"Using credentials: UserName={omero_username} Password={len(omero_password)*'*'}")
    with BlitzGateway(omero_username, omero_password, host=omero_host) as connection:
        update_service = connection.getUpdateService()
        result = connection.connect()
        #print(f"Connect: {result}")
        _username = connection.getUser().getName()
        _fullname = connection.getUser().getFullName()
        #print(f"Connected as: UserName={_username} FullName={_fullname}")
        group_id = -1
        #group_id = 409
        #print(f"SERVICE_OPTS.setOmeroGroup: {group_id}")
        connection.SERVICE_OPTS.setOmeroGroup(group_id)

        #print(f"Get image: Id={image_id}")
        image = connection.getObject("Image", image_id)
        #print(f"IMAGE: Id={image.id} Name={image.name}")
        #print(f"GROUP: Id={image.details.group.id.val} Name={image.details.group.name.val}")
        roi = create_roi(image, [polygon])
        #roi = create_roi(image, points)
        #roi.textValue = rstring("test-Polygon")
        update_service.saveObject(roi)

def collect_ROIs_from_OMERO(omero_username, omero_password, omero_host, omero_image_id):
    ROIs = []
    log.info(f"Connecting to OMERO at {omero_host}")
    with BlitzGateway(omero_username, omero_password, host=omero_host, port=4064, secure=True) as conn:
        # search image
        log.info(f"Looking for ImageId {omero_image_id}")
        conn.SERVICE_OPTS.setOmeroGroup('-1')
        image = conn.getObject("Image", omero_image_id)
        # set group for image
        log.info(f"Found image id={image.id} name='{image.name}'")
        log.info(f"Found image in group id={image.details.group.id.val} name='{image.details.group.name.val}'")
        
        log.info(f"Storing rendered thumbnail in memory for QC")
        img_data = image.getThumbnail() #tiny preview image
        rendered_thumb = Image.open(io.BytesIO(img_data))
       
        group_id = image.details.group.id #check group id
        conn.setGroupForSession(group_id.val)
        # get image ROIs
        roi_service = conn.getRoiService()
        log.info("Retrieving ROIs")
        result = roi_service.findByImage(image.id, None)
        #result has property roi - for one given image id
        for roi in result.rois:

            primary_shape = roi.getPrimaryShape()
            name = primary_shape.getTextValue().val
            #if ROI name is empty - renamed to "Non_labelled", makes sure first letter is capital, if additional csv provided - unified all different ROIs name into one
            #print(name)
            #, separates x and y and  space separates points
            try:
                points = [(lambda xy : list(map(float,xy.split(","))))(xy) for xy in primary_shape.getPoints().val.split(" ")]
                ROIs.append({
                "name": name,
                "points": points
                })
            except:
                if primary_shape.__class__.__name__ == 'RectangleI':
                    points = get_corners_rectangle(primary_shape)
                    ROIs.append({
                    "name": name,
                    "points": points
                    })
                else:
                    pass
            
            log.debug(f"Found ROI id={roi.id.val} name='{name}' type={primary_shape.__class__.__name__}")

    log.info(f"Found {len(ROIs)} ROIs in total")
    return ROIs, image

def array_to_string(ar):
    ss = ""
    for i in range(len(ar)):
        ss = ss + str(int(ar[i][0])) + ',' + str(int(ar[i][1])) + ' '
    return ss

def get_OMERO_credentials():
    logging.basicConfig(format='%(asctime)s %(message)s')    
    log.setLevel(logging.DEBUG)
    #omero_host = "wsi-omero-prod-02.internal.sanger.ac.uk"
    omero_username = input("Username$")
    omero_password = getpass("Password$")
    return omero_username, omero_password

def contstruct_polygon(roi):
    polygon = omero.model.PolygonI()
    polygon.fillColor = rint(rgba_to_int(255, 0, 255, 50))
    polygon.strokeColor = rint(rgba_to_int(255, 255, 0))
    polygon.strokeWidth = omero.model.LengthI(10, UnitsLength.PIXEL)
    points = array_to_string(roi['points'])
    polygon.points = rstring(points)
    polygon.textValue = rstring(roi['name'])
    return polygon

def main(csv_path):
    table_input = pd.read_csv(csv_path)
    #initialization
    print('Please provide credentials for OMERO server 1:')
    omero_host_1 = "wsi-omero-prod-02.internal.sanger.ac.uk"
    omero_username_1, omero_password_1 = get_OMERO_credentials()
    
    print('Please provide credentials for OMERO server 2:')
    omero_host_2 = "wsi-omero-prod-02.internal.sanger.ac.uk"
    omero_username_2, omero_password_2 = get_OMERO_credentials()
    
    for i in range(table_input.shape[0]):
        print(i)
        omero_id_in = table_input['omero_id_1'][i]
        omero_id_out = table_input['omero_id_2'][i]
        ROIs, image = collect_ROIs_from_OMERO(omero_username_1, omero_password_1, omero_host_1, omero_id_in)
        for roi in ROIs:
            pol = contstruct_polygon(roi)
            write_one_roi(omero_username_2, omero_password_2, omero_host_2, pol, omero_id_out)
   

if __name__ == "__main__":
    fire.Fire(main) 
