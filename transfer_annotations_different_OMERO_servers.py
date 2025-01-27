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

def write_one_roi(omero_username, omero_password, omero_host, polygon, image_id, omero_group_id, omero_user_id):
    
    #print(f"Connecting to {omero_srv} in port {omero_port}")
    #print(f"Using credentials: UserName={omero_username} Password={len(omero_password)*'*'}")
    with BlitzGateway(omero_username, omero_password, host=omero_host, secure = True) as connection:
        update_service = connection.getUpdateService()
        result = connection.connect()
        if omero_group_id != -1:
            connection.SERVICE_OPTS.setOmeroGroup(omero_group_id)
            connection.SERVICE_OPTS.setOmeroUser(omero_user_id)
        else:
            connection.SERVICE_OPTS.setOmeroGroup('-1')
        #print(f"SERVICE_OPTS.setOmeroGroup: {group_id}")
        #print(f"Get image: Id={image_id}")
        image = connection.getObject("Image", image_id)
        #print(f"IMAGE: Id={image.id} Name={image.name}")
        #print(f"GROUP: Id={image.details.group.id.val} Name={image.details.group.name.val}")
        roi = create_roi(image, [polygon])
        if omero_group_id != -1:
            update_service.saveObject(roi, connection.SERVICE_OPTS)
        else:
            update_service.saveObject(roi)

def collect_ROIs_from_OMERO(omero_username, omero_password, omero_host, omero_port, omero_image_id):
    ROIs = []
    log.info(f"Connecting to OMERO at {omero_host}")
    with BlitzGateway(omero_username, omero_password, host=omero_host, port=omero_port, secure=True) as conn:
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
            stroke_color = roi.getPrimaryShape().getStrokeColor()
            fill_color = roi.getPrimaryShape().getFillColor()
            stroke_width = roi.getPrimaryShape().getStrokeWidth()
            stroke_dash =  roi.getPrimaryShape().getStrokeDashArray()
            #if ROI name is empty - renamed to "Non_labelled", makes sure first letter is capital, if additional csv provided - unified all different ROIs name into one
            #print(name)
            #, separates x and y and  space separates points
            
            
           
            try:
                points = []
                for xy in primary_shape.getPoints().val.split(" "):
                    if len(xy)>3:
                        points.append([float(xy.split(",")[0]), float(xy.split(",")[1])])

                    #points = [(lambda xy : list(map(float,xy.split(","))))(xy) for xy in primary_shape.getPoints().val.split(" ")]
                ROIs.append({
                "name": name,
                "points": points,
                "stroke_color": stroke_color,
                "stroke_width": stroke_width,
                "fill_color": fill_color,
                "stroke_dash": stroke_dash,    
                })
            except:
                if primary_shape.__class__.__name__ == 'RectangleI':
                    points = get_corners_rectangle(primary_shape)
                    ROIs.append({
                    "name": name,
                    "points": points,
                    "stroke_color": stroke_color,
                    "stroke_width": stroke_width,
                    "fill_color": fill_color,
                    "stroke_dash": stroke_dash,  
                    })
                else:
                    pass
                
            log.debug(f"Found ROI id={roi.id.val} name='{name}' type={primary_shape.__class__.__name__}")

    log.info(f"Found {len(ROIs)} ROIs in total")
    return ROIs, image

def get_corners_rectangle(primary_shape):
    x0 = primary_shape.getX()._val; y0 = primary_shape.getY()._val
    w = primary_shape.getWidth()._val; h = primary_shape.getHeight()._val
    return [(x0, y0), (x0+w,y0), (x0+w,y0+h), (x0,y0+h)]

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

def contstruct_polygon(roi, stroke_width_default = 5):
    #print(roi)
    polygon = omero.model.PolygonI()
    polygon.fillColor = roi['fill_color']
    polygon.strokeColor = roi['stroke_color']
    try:
        polygon.strokeDashArray = roi['stroke_dash']
    except:
        pass
    if isinstance(roi['stroke_width'], (int, float)):
        polygon.strokeWidth = roi['stroke_width']
    else:
        polygon.strokeWidth = omero.model.LengthI(stroke_width_default, UnitsLength.PIXEL)
        
    points = array_to_string(roi['points'])
    polygon.points = rstring(points)
    polygon.textValue = rstring(roi['name'])
    return polygon

def main(csv_path):
    table_input = pd.read_csv(csv_path)
    #initialization
    print('Please provide credentials for OMERO server 1:')
    omero_host_1 = "wsi-omero-prod-02.internal.sanger.ac.uk"
    omero_port_1 = 4064
    omero_username_1, omero_password_1 = get_OMERO_credentials()
    
    print('Please provide credentials for OMERO server 2:')
    omero_host_2 = "wsi-omero-prod-02.internal.sanger.ac.uk" #"dtomero.sdsc.edu"
    omero_username_2, omero_password_2 = get_OMERO_credentials()
    
    omero_2_group = -1
    omero_2_user_id = None
    #fill those values only if you have permission issues of running write_roi function
    #omero_2_group = 409
    #omero_2_user_id = 302
    
    
    
    for i in range(table_input.shape[0]):
        
        omero_id_in = table_input['omero_id_1'][i]
        omero_id_out = table_input['omero_id_2'][i]
        ROIs, image = collect_ROIs_from_OMERO(omero_username_1, omero_password_1, omero_host_1, omero_port_1, omero_id_in)
        print(str(i) + '/' + str(table_input.shape[0]))
        for roi in ROIs:
            pol = contstruct_polygon(roi)
            write_one_roi(omero_username_2, omero_password_2, omero_host_2, pol, omero_id_out, omero_2_group, omero_2_user_id)
   

if __name__ == "__main__":
    fire.Fire(main) 
