import os
import sys
import gc
import warnings
import time
import csv
import pathlib
import numpy as np

from osgeo import gdal
from osgeo import osr
import wradlib as wrl

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QThread, pyqtSignal

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import ListedColormap,BoundaryNorm

import tkinter as Tk
from tkinter import PhotoImage
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk as NavigationToolbar2TkAgg

beam_block_doc = {"type": "np.array"}
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['axes.titlepad'] = 20
warnings.filterwarnings('ignore')

def check_path(fpath):
	if not os.path.isdir(fpath):
		os.mkdir(fpath)

def ELEVS_CMAP():
	bounds=np.array([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0])
	bounds = bounds+0.001
	cmap=ListedColormap([
		'#00ffff','#009efa','#0050f7','#002693','#0000a0',
		'#00ff00','#02c802','#008900','#006400','#003c00',
		'#ffff00','#e7bd00','#ff9400','#ff4800','#ff0000',
		'#d70000','#a50000','#ff00ff','#bf00bf','#7800be','#3d0079'])
	cmap.set_over('#000000')
	cmap.set_under('#ffffff')
	norm=BoundaryNorm(bounds, cmap.N)
	return cmap,norm,bounds 

def HEIGHT_CMAP():
	bounds=np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.5, 4.0, 4.5, 5.0, 7.0, 10.0])
	#bounds = bounds+0.001
	cmap=ListedColormap([
		'#00ffff','#009efa','#0050f7','#002693','#0000a0',
		'#00ff00','#02c802','#008900','#006400','#003c00',
		'#ffff00','#e7bd00','#ff9400','#ff4800','#ff0000',
		'#d70000','#a50000','#ff00ff','#bf00bf','#7800be','#3d0079'])
	cmap.set_over('#000000')
	cmap.set_under('#ffffff')
	norm=BoundaryNorm(bounds, cmap.N)
	return cmap,norm,bounds 

# def convert_height(data, range_, radar_height):
# 	sin_azi = np.sin(data*0.0174532925)
# 	for i in range(data.shape[0]):
# 		H = range_*sin_azi[i]+(range_**2/(2*((4/3)*6370000)))+radar_height
# 		hei__ = np.where(data!=9999, H, -9999)
# 	return hei__

def convert_height(data, range_, radar_height):
	hei__ = []
	sin_azi = np.sin(data*0.0174532925)
	for i in range(data.shape[0]):
		H = range_*sin_azi[i]+(range_**2/(2*((4/3)*6370000)))+radar_height
		H = np.where(data[i]!=9999, H, -9999)
		hei__.append(H)
	hei__ = np.array(hei__)
	return hei__

class create_radar:
	def __init__(self, radar_name, lon, lat, alt, nrays, nbins, elevs, range_res, beam_width):
		self.radar_name = radar_name
		self.site_coord = (lon, lat, alt)
		self.nrays = nrays
		self.nbins = nbins
		if (len(elevs)>1):
			elevs = np.sort(elevs, axis=- 1)
		self.elevs = elevs
		self.range_res = float(range_res)
		self.beam_width = beam_width

		dis_res = (self.range_res*self.nbins)/100000+0.3

		self.com_grid={}
		self.com_grid['lonmin']=self.site_coord[0]-dis_res
		self.com_grid['latmin']=self.site_coord[1]-dis_res
		self.com_grid['lonmax']=self.site_coord[0]+dis_res
		self.com_grid['latmax']=self.site_coord[1]+dis_res
		self.com_grid['reslu_horiz']=600
		self.com_grid['reslu_vert']=600

	def __eq__(self, other):
		if (isinstance(other, create_radar)):
			return self.radar_name == other.radar_name or self.site_coord == other.site_coord

def convert_ppi_composite(radar_obj, data):
	com_grid={}
	com_grid['lonmin']=97
	com_grid['lonmax']=115
	com_grid['latmin']=7.2
	com_grid['latmax']=25.2
	com_grid['reslu_horiz']=1980
	com_grid['reslu_vert']=1980

	lon = np.linspace(com_grid['lonmin'], com_grid['lonmax'], com_grid['reslu_horiz'])
	lat = np.linspace(com_grid['latmin'], com_grid['latmax'], com_grid['reslu_vert'])
	grid_xy = np.meshgrid(lon, lat)
	grid_xy = np.vstack((grid_xy[0].ravel(), grid_xy[1].ravel())).transpose()

	data = data.reshape(1, int(radar_obj.nrays), int(radar_obj.nbins))
	azimuths = np.arange(0., 360., 1.)
	ranges = np.arange(int(radar_obj.nbins)) * radar_obj.range_res
	R = ((int(radar_obj.nbins)) * radar_obj.range_res)/110000.
	radar_location = radar_obj.site_coord
	polargrid = np.meshgrid(ranges, azimuths)
	coords, rad = wrl.georef.spherical_to_xyz(polargrid[0], polargrid[1],
									radar_obj.elevs[0], radar_location)
	gk3 = wrl.georef.epsg_to_osr(4326)
	gk3_coords = wrl.georef.reproject(coords, projection_source=rad,
									  projection_target=gk3)
	x = gk3_coords[..., 0]
	y = gk3_coords[..., 1]
	xy=np.concatenate([x[0].ravel()[:,None],y[0].ravel()[:,None]], axis=1)
	gridded = wrl.comp.togrid(xy, grid_xy, R, np.array([x.mean(), y.mean()]), np.array(data).ravel(), wrl.ipol.Idw)
	del com_grid, xy, x, y, lon, lat, radar_obj
	del gk3, gk3_coords, coords, rad, polargrid
	del radar_location, R, ranges, azimuths, data, grid_xy
	gc.collect()
	return gridded

def convert_ppi_single(radar_obj, data):
	com_grid = radar_obj.com_grid
	lon = np.linspace(com_grid['lonmin'], com_grid['lonmax'], com_grid['reslu_horiz'])
	lat = np.linspace(com_grid['latmin'], com_grid['latmax'], com_grid['reslu_vert'])
	grid_xy = np.meshgrid(lon, lat)
	grid_xy = np.vstack((grid_xy[0].ravel(), grid_xy[1].ravel())).transpose()

	data = data.reshape(1, int(radar_obj.nrays), int(radar_obj.nbins))
	azimuths = np.arange(0., 360., 1.)
	ranges = np.arange(int(radar_obj.nbins)) * radar_obj.range_res
	R = ((int(radar_obj.nbins)) * radar_obj.range_res)/110000.
	radar_location = radar_obj.site_coord
	polargrid = np.meshgrid(ranges, azimuths)
	coords, rad = wrl.georef.spherical_to_xyz(polargrid[0], polargrid[1],
									radar_obj.elevs[0], radar_location)
	gk3 = wrl.georef.epsg_to_osr(4326)
	gk3_coords = wrl.georef.reproject(coords, projection_source=rad,
									  projection_target=gk3)
	x = gk3_coords[..., 0]
	y = gk3_coords[..., 1]
	xy=np.concatenate([x[0].ravel()[:,None],y[0].ravel()[:,None]], axis=1)
	gridded = wrl.comp.togrid(xy, grid_xy, R, np.array([x.mean(), y.mean()]), np.array(data).ravel(), wrl.ipol.Idw)
	gridded = gridded.reshape(com_grid['reslu_horiz'],com_grid['reslu_vert'])
	del com_grid, xy, x, y, lon, lat, radar_obj
	del gk3, gk3_coords, coords, rad, polargrid
	del radar_location, R, ranges, azimuths, data, grid_xy
	gc.collect()
	return gridded

def create_cbb_elev_single(q_thread, radar_obj)->beam_block_doc:
	q_thread.signal.emit(0)
	ds = wrl.io.open_raster("gt30e100n40.tif")
	
	rastervalues, rastercoords, proj = wrl.georef.extract_raster_dataset(ds, nodata=-32768.)
	data_ = np.zeros((radar_obj.nrays, radar_obj.nbins))
	data_ = np.where(data_==0, 9999, 9999)
	num=0
	for el in range(len(radar_obj.elevs)-1, -1, -1):
		r = np.arange(radar_obj.nbins) * radar_obj.range_res
		q_thread.signal.emit(int(int(100/len(radar_obj.elevs))/6)+int(100/len(radar_obj.elevs))*num)
		beamradius = wrl.util.half_power_radius(r, radar_obj.beam_width)
		q_thread.signal.emit(int(int(100/len(radar_obj.elevs))/5)+int(100/len(radar_obj.elevs))*num)
		coord = wrl.georef.sweep_centroids(radar_obj.nrays, radar_obj.range_res, radar_obj.nbins, radar_obj.elevs[el])
		q_thread.signal.emit(int(int(100/len(radar_obj.elevs))/4)+int(100/len(radar_obj.elevs))*num)
		coords = wrl.georef.spherical_to_proj(coord[..., 0],
											  coord[..., 1],
											  coord[..., 2], radar_obj.site_coord)
		q_thread.signal.emit(int(int(100/len(radar_obj.elevs))/3)+int(100/len(radar_obj.elevs))*num)
		lon = coords[..., 0]
		lat = coords[..., 1]
		alt = coords[..., 2]
		polcoords = coords[..., :2]
		rlimits = (lon.min(), lat.min(), lon.max(), lat.max())
		
		q_thread.signal.emit(int(int(100/len(radar_obj.elevs))/2)+int(100/len(radar_obj.elevs))*num)
		ind = wrl.util.find_bbox_indices(rastercoords, rlimits)
		rastercoord = rastercoords[ind[1]:ind[3], ind[0]:ind[2], ...]
		rastervalue = rastervalues[ind[1]:ind[3], ind[0]:ind[2]]
		q_thread.signal.emit(int(int(100/len(radar_obj.elevs))/1.6)+int(100/len(radar_obj.elevs))*num)
		polarvalues = wrl.ipol.cart_to_irregular_spline(rastercoord, rastervalue,
													 polcoords, order=3,
													 prefilter=False)
		PBB = wrl.qual.beam_block_frac(polarvalues, alt, beamradius)
		q_thread.signal.emit(int(int(100/len(radar_obj.elevs))/1.2)+int(100/len(radar_obj.elevs))*num)
		PBB = np.ma.masked_invalid(PBB)
		CBB = wrl.qual.cum_beam_block_frac(PBB)
		data_ = np.where(CBB<0.2, radar_obj.elevs[el], data_)
		q_thread.signal.emit(int(100/len(radar_obj.elevs))+int(100/len(radar_obj.elevs))*num)
		num+=1
	del q_thread, radar_obj
	del ds, rastervalues, rastercoords, proj
	del num, el, r, beamradius, coord, coords
	del lon, lat, alt, polcoords, rlimits, ind
	del rastercoord, rastervalue, polarvalues
	del PBB, CBB
	gc.collect()
	return data_

def create_cbb_hei_single_ahihi(q_thread, radar_obj)->beam_block_doc:
	q_thread.signal.emit(0)
	ds = wrl.io.open_raster("gt30e100n40.tif")
	
	rastervalues, rastercoords, proj = wrl.georef.extract_raster_dataset(ds, nodata=-32768.)
	#data_ = np.zeros((radar_obj.nrays, radar_obj.nbins))
	#data_ = np.where(data_==0, 9999, 9999)
	beam_alt = np.zeros((radar_obj.nrays, radar_obj.nbins))
	beam_alt = np.where(beam_alt==0, 9999, 9999)
	num=0
	for el in range(len(radar_obj.elevs)-1, -1, -1):
		r = np.arange(radar_obj.nbins) * radar_obj.range_res
		q_thread.signal.emit(int(int(100/len(radar_obj.elevs))/6)+int(100/len(radar_obj.elevs))*num)
		beamradius = wrl.util.half_power_radius(r, radar_obj.beam_width)
		q_thread.signal.emit(int(int(100/len(radar_obj.elevs))/5)+int(100/len(radar_obj.elevs))*num)
		coord = wrl.georef.sweep_centroids(radar_obj.nrays, radar_obj.range_res, radar_obj.nbins, radar_obj.elevs[el])
		q_thread.signal.emit(int(int(100/len(radar_obj.elevs))/4)+int(100/len(radar_obj.elevs))*num)
		coords = wrl.georef.spherical_to_proj(coord[..., 0],
											  coord[..., 1],
											  coord[..., 2], radar_obj.site_coord)
		q_thread.signal.emit(int(int(100/len(radar_obj.elevs))/3)+int(100/len(radar_obj.elevs))*num)
		lon = coords[..., 0]
		lat = coords[..., 1]
		alt = coords[..., 2]
		polcoords = coords[..., :2]
		rlimits = (lon.min(), lat.min(), lon.max(), lat.max())
		
		q_thread.signal.emit(int(int(100/len(radar_obj.elevs))/2)+int(100/len(radar_obj.elevs))*num)
		ind = wrl.util.find_bbox_indices(rastercoords, rlimits)
		rastercoord = rastercoords[ind[1]:ind[3], ind[0]:ind[2], ...]
		rastervalue = rastervalues[ind[1]:ind[3], ind[0]:ind[2]]
		q_thread.signal.emit(int(int(100/len(radar_obj.elevs))/1.6)+int(100/len(radar_obj.elevs))*num)
		polarvalues = wrl.ipol.cart_to_irregular_spline(rastercoord, rastervalue,
													 polcoords, order=3,
													 prefilter=False)
		PBB = wrl.qual.beam_block_frac(polarvalues, alt, beamradius)
		q_thread.signal.emit(int(int(100/len(radar_obj.elevs))/1.2)+int(100/len(radar_obj.elevs))*num)
		PBB = np.ma.masked_invalid(PBB)
		CBB = wrl.qual.cum_beam_block_frac(PBB)
		# print (alt.shape)
		# print (CBB.shape)
		beam_alt = np.where(CBB<0.2, alt, beam_alt)
		#data_ = np.where(CBB<0.2, radar_obj.elevs[el], data_)
		q_thread.signal.emit(int(100/len(radar_obj.elevs))+int(100/len(radar_obj.elevs))*num)
		num+=1
	return beam_alt

def create_cbb_hei_single(q_thread, radar_obj)->beam_block_doc:
	q_thread.signal.emit(0)
	ds = wrl.io.open_raster("gt30e100n40.tif")
	
	rastervalues, rastercoords, proj = wrl.georef.extract_raster_dataset(ds, nodata=-32768.)
	data_ = np.zeros((radar_obj.nrays, radar_obj.nbins))
	data_ = np.where(data_==0, 9999, 9999)
	num=0
	for el in range(len(radar_obj.elevs)-1, -1, -1):
		r = np.arange(radar_obj.nbins) * radar_obj.range_res
		q_thread.signal.emit(int(int(100/len(radar_obj.elevs))/6)+int(100/len(radar_obj.elevs))*num)
		beamradius = wrl.util.half_power_radius(r, radar_obj.beam_width)
		q_thread.signal.emit(int(int(100/len(radar_obj.elevs))/5)+int(100/len(radar_obj.elevs))*num)
		coord = wrl.georef.sweep_centroids(radar_obj.nrays, radar_obj.range_res, radar_obj.nbins, radar_obj.elevs[el])
		q_thread.signal.emit(int(int(100/len(radar_obj.elevs))/4)+int(100/len(radar_obj.elevs))*num)
		coords = wrl.georef.spherical_to_proj(coord[..., 0],
											  coord[..., 1],
											  coord[..., 2], radar_obj.site_coord)
		q_thread.signal.emit(int(int(100/len(radar_obj.elevs))/3)+int(100/len(radar_obj.elevs))*num)
		lon = coords[..., 0]
		lat = coords[..., 1]
		alt = coords[..., 2]
		polcoords = coords[..., :2]
		rlimits = (lon.min(), lat.min(), lon.max(), lat.max())
		
		q_thread.signal.emit(int(int(100/len(radar_obj.elevs))/2)+int(100/len(radar_obj.elevs))*num)
		ind = wrl.util.find_bbox_indices(rastercoords, rlimits)
		rastercoord = rastercoords[ind[1]:ind[3], ind[0]:ind[2], ...]
		rastervalue = rastervalues[ind[1]:ind[3], ind[0]:ind[2]]
		q_thread.signal.emit(int(int(100/len(radar_obj.elevs))/1.6)+int(100/len(radar_obj.elevs))*num)
		polarvalues = wrl.ipol.cart_to_irregular_spline(rastercoord, rastervalue,
													 polcoords, order=3,
													 prefilter=False)
		PBB = wrl.qual.beam_block_frac(polarvalues, alt, beamradius)
		q_thread.signal.emit(int(int(100/len(radar_obj.elevs))/1.2)+int(100/len(radar_obj.elevs))*num)
		PBB = np.ma.masked_invalid(PBB)
		CBB = wrl.qual.cum_beam_block_frac(PBB)
		data_ = np.where(CBB<0.2, radar_obj.elevs[el], data_)
		q_thread.signal.emit(int(100/len(radar_obj.elevs))+int(100/len(radar_obj.elevs))*num)
		num+=1
	del q_thread, radar_obj
	del ds, rastervalues, rastercoords, proj
	del num, el, r, beamradius, coord, coords
	del lon, lat, alt, polcoords, rlimits, ind
	del rastercoord, rastervalue, polarvalues
	del PBB, CBB
	gc.collect()
	return data_

def create_cbb_elev_composite(q_thread, radar_obj, total_radar, current_single)->beam_block_doc:
	q_thread.signal.emit(current_single)
	ds = wrl.io.open_raster("gt30e100n40.tif")
	rastervalues, rastercoords, proj = wrl.georef.extract_raster_dataset(ds, nodata=-32768.)
	data_ = np.zeros((radar_obj.nrays, radar_obj.nbins))
	data_ = np.where(data_==0, 9999, 9999)
	num=0
	for el in range(len(radar_obj.elevs)-1, -1, -1):
		r = np.arange(radar_obj.nbins) * radar_obj.range_res

		q_thread.signal.emit(int((int(int(100/len(radar_obj.elevs))/6)+int(100/len(radar_obj.elevs))*num)/total_radar)+current_single)
		beamradius = wrl.util.half_power_radius(r, radar_obj.beam_width)
		q_thread.signal.emit(int((int(int(100/len(radar_obj.elevs))/5)+int(100/len(radar_obj.elevs))*num)/total_radar)+current_single)
		coord = wrl.georef.sweep_centroids(radar_obj.nrays, radar_obj.range_res, radar_obj.nbins, radar_obj.elevs[el])
		q_thread.signal.emit(int((int(int(100/len(radar_obj.elevs))/4)+int(100/len(radar_obj.elevs))*num)/total_radar)+current_single)
		coords = wrl.georef.spherical_to_proj(coord[..., 0],
											  coord[..., 1],
											  coord[..., 2], radar_obj.site_coord)
		q_thread.signal.emit(int((int(int(100/len(radar_obj.elevs))/3)+int(100/len(radar_obj.elevs))*num)/total_radar)+current_single)
		lon = coords[..., 0]
		lat = coords[..., 1]
		alt = coords[..., 2]
		polcoords = coords[..., :2]
		rlimits = (lon.min(), lat.min(), lon.max(), lat.max())
		
		q_thread.signal.emit(int((int(int(100/len(radar_obj.elevs))/2)+int(100/len(radar_obj.elevs))*num)/total_radar)+current_single)
		ind = wrl.util.find_bbox_indices(rastercoords, rlimits)
		rastercoord = rastercoords[ind[1]:ind[3], ind[0]:ind[2], ...]
		rastervalue = rastervalues[ind[1]:ind[3], ind[0]:ind[2]]
		q_thread.signal.emit(int((int(int(100/len(radar_obj.elevs))/1.6)+int(100/len(radar_obj.elevs))*num)/total_radar)+current_single)
		polarvalues = wrl.ipol.cart_to_irregular_spline(rastercoord, rastervalue,
													 polcoords, order=3,
													 prefilter=False)
		PBB = wrl.qual.beam_block_frac(polarvalues, alt, beamradius)
		q_thread.signal.emit(int((int(int(100/len(radar_obj.elevs))/1.2)+int(100/len(radar_obj.elevs))*num)/total_radar)+current_single)
		PBB = np.ma.masked_invalid(PBB)
		CBB = wrl.qual.cum_beam_block_frac(PBB)
		data_ = np.where(CBB<0.2, radar_obj.elevs[el], data_)
		q_thread.signal.emit(int((int(100/len(radar_obj.elevs))+int(100/len(radar_obj.elevs))*num)/total_radar)+current_single)
		num+=1
	del q_thread, radar_obj, total_radar, current_single
	del ds, rastervalues, rastercoords, proj
	del num, el, r, beamradius, coord, coords
	del lon, lat, alt, polcoords, rlimits, ind
	del rastercoord, rastervalue, polarvalues
	del PBB, CBB
	gc.collect()
	return data_

def create_cbb_hei_composite(q_thread, radar_obj, total_radar, current_single)->beam_block_doc:
	q_thread.signal.emit(current_single)
	ds = wrl.io.open_raster("gt30e100n40.tif")
	rastervalues, rastercoords, proj = wrl.georef.extract_raster_dataset(ds, nodata=-32768.)
	data_ = np.zeros((radar_obj.nrays, radar_obj.nbins))
	data_ = np.where(data_==0, 9999, 9999)
	num=0
	for el in range(len(radar_obj.elevs)-1, -1, -1):
		r = np.arange(radar_obj.nbins) * radar_obj.range_res
		q_thread.signal.emit(int((int(int(100/len(radar_obj.elevs))/6)+int(100/len(radar_obj.elevs))*num)/total_radar)+current_single)
		beamradius = wrl.util.half_power_radius(r, radar_obj.beam_width)
		q_thread.signal.emit(int((int(int(100/len(radar_obj.elevs))/5)+int(100/len(radar_obj.elevs))*num)/total_radar)+current_single)
		coord = wrl.georef.sweep_centroids(radar_obj.nrays, radar_obj.range_res, radar_obj.nbins, radar_obj.elevs[el])
		q_thread.signal.emit(int((int(int(100/len(radar_obj.elevs))/4)+int(100/len(radar_obj.elevs))*num)/total_radar)+current_single)
		coords = wrl.georef.spherical_to_proj(coord[..., 0],
											  coord[..., 1],
											  coord[..., 2], radar_obj.site_coord)
		q_thread.signal.emit(int((int(int(100/len(radar_obj.elevs))/3)+int(100/len(radar_obj.elevs))*num)/total_radar)+current_single)
		lon = coords[..., 0]
		lat = coords[..., 1]
		alt = coords[..., 2]
		polcoords = coords[..., :2]
		rlimits = (lon.min(), lat.min(), lon.max(), lat.max())
		
		q_thread.signal.emit(int((int(int(100/len(radar_obj.elevs))/2)+int(100/len(radar_obj.elevs))*num)/total_radar)+current_single)
		ind = wrl.util.find_bbox_indices(rastercoords, rlimits)
		rastercoord = rastercoords[ind[1]:ind[3], ind[0]:ind[2], ...]
		rastervalue = rastervalues[ind[1]:ind[3], ind[0]:ind[2]]
		q_thread.signal.emit(int((int(int(100/len(radar_obj.elevs))/1.6)+int(100/len(radar_obj.elevs))*num)/total_radar)+current_single)
		polarvalues = wrl.ipol.cart_to_irregular_spline(rastercoord, rastervalue,
													 polcoords, order=3,
													 prefilter=False)
		PBB = wrl.qual.beam_block_frac(polarvalues, alt, beamradius)
		q_thread.signal.emit(int((int(int(100/len(radar_obj.elevs))/1.2)+int(100/len(radar_obj.elevs))*num)/total_radar)+current_single)
		PBB = np.ma.masked_invalid(PBB)
		CBB = wrl.qual.cum_beam_block_frac(PBB)
		data_ = np.where(CBB<0.2, radar_obj.elevs[el], data_)
		q_thread.signal.emit(int((int(100/len(radar_obj.elevs))+int(100/len(radar_obj.elevs))*num)/total_radar)+current_single)
		num+=1
	del q_thread, radar_obj, total_radar, current_single
	del ds, rastervalues, rastercoords, proj
	del num, el, r, beamradius, coord, coords
	del lon, lat, alt, polcoords, rlimits, ind
	del rastercoord, rastervalue, polarvalues
	del PBB, CBB
	gc.collect()
	return data_

def save_radar_img(data, radar_obj=None, gr="COM", cm="ELE", str_com="", link_img="img/"):
	fig = plt.figure(figsize=(8,8))

	if (gr=="COM"):
		map = Basemap(llcrnrlon=97,llcrnrlat=7.2,urcrnrlon=115,urcrnrlat=25.2, epsg=4326, ellps='WGS84')
		img_name = link_img + "Composite_Radar_" + cm + str_com + ".png"
	else:
		map = Basemap(llcrnrlon=(radar_obj.com_grid["lonmin"]),llcrnrlat=(radar_obj.com_grid["latmin"]),urcrnrlon=(radar_obj.com_grid["lonmax"]),urcrnrlat=(radar_obj.com_grid["latmax"]), epsg=4326, ellps='WGS84')
		img_name = link_img + str(radar_obj.radar_name) + "_" + cm + "_" + str(radar_obj.site_coord[0]) + "_" + str(radar_obj.site_coord[1]) + "_" + str(radar_obj.site_coord[2]) + "_" + str(len(radar_obj.elevs)) + "_" + str(radar_obj.beam_width) + ".png"
	if (cm=="ELE"):
		cmap,norm,bounds = ELEVS_CMAP()
		#bounds = bounds-0.001
		im = map.imshow(data, origin="lower",cmap=cmap,norm=norm)
	else:
		cmap,norm,bounds = HEIGHT_CMAP()
		im = map.imshow(data/1000, origin="lower",cmap=cmap,norm=norm)

	map.readshapefile('shapefile/sss','sss', linewidth=0.5, color='k',  default_encoding='utf-8')
	map.drawparallels(np.arange(-90.0, 90.0, 2), linewidth=0.4, dashes=[4, 4], color='blue', labels=[True,False,False,True], fmt='%g',size=6)
	map.drawmeridians(np.arange(0.0, 360.0, 2), linewidth=0.4, dashes=[4, 4], color='blue', labels=[True,False,False,True], fmt='%g',size=6)
	# plt.text(112.8, 9.5, "Q. Trường Sa", ha="center", family='sans-serif', size=8, color="brown")
	# plt.text(111.5, 17.0, "Q. Hoàng Sa", ha="center", family='sans-serif', size=8, color="brown")
	cbar = plt.colorbar(cmap=cmap, ticks=bounds, orientation='vertical', 
				shrink=2.4, aspect=36, fraction=0.02, pad=0.08,
				extend='both', extendfrac=None, extendrect=False)
	cbar.set_label("Elevs.[°]", labelpad=-20, y=1.05, rotation=0)
	plt.xlabel('Longitude [deg]', labelpad=20.0)
	plt.ylabel('Latitude [deg]', labelpad=32.0)
	plt.title('Single radar elevations')	
	plt.savefig(img_name, dpi=400, bbox_inches='tight',pad_inches=0.2)
	plt.clf()
	plt.close()
	del data, radar_obj, gr, cm, str_com, link_img
	del fig, map, im, img_name
	del cmap, norm, bounds, cbar
	gc.collect()

class SingleThread_Elevs(QThread):
	signal = pyqtSignal('PyQt_PyObject')
	def __init__(self, radar_obj, link_img):
		QThread.__init__(self)
		self.radar_obj = radar_obj
		self.link_img = link_img

	def run(self):
		data_ = create_cbb_elev_single(self, self.radar_obj)
		data_ = convert_ppi_single(self.radar_obj, data_)
		#save_radar_img(data_, self.radar_obj, gr="single", cm="ele", link_img=self.link_img)
		self.signal.emit(100)
		self.signal.emit(data_)
		del self.radar_obj, self.link_img
		del data_
		gc.collect()

		try:
			self.stop()
			gc.collect()
		except:
			pass

	def stop(self):
		#self.exit(0)
		self.exec()
		self.wait()

class SingleThread_Height(QThread):
	signal = pyqtSignal('PyQt_PyObject')
	def __init__(self, radar_obj, link_img):
		QThread.__init__(self)
		self.radar_obj = radar_obj
		self.link_img = link_img

	def run(self):
		range__ = np.arange(self.radar_obj.nbins) * self.radar_obj.range_res
		#data_ = create_cbb_single(self, self.radar_obj)
		data_ = create_cbb_hei_single(self, self.radar_obj)
		data_ = convert_height(data_, range__, self.radar_obj.site_coord[2])
		data_ = np.where(data_<0,0,data_)
		data_ = convert_ppi_single(self.radar_obj, data_)
		#save_radar_img(data_, self.radar_obj, gr="single", cm="hei", link_img=self.link_img)
		self.signal.emit(100)
		self.signal.emit(data_)
		del self.radar_obj, self.link_img
		del data_, range__
		gc.collect()
		try:
			self.stop()
			gc.collect()
		except:
			pass

	def stop(self):
		#self.exit(0)
		self.exec()
		self.wait()

class CompositeThread_Elevs(QThread):
	signal = pyqtSignal('PyQt_PyObject')
	def __init__(self, com_radar, link_img):
		QThread.__init__(self)
		self.com_radar = com_radar
		self.link_img = link_img

	def run(self):
		data_com = np.zeros((1980*1980))
		data_com = np.where(data_com==0, -9999, -9999)
		for num, radar_i in enumerate(self.com_radar):
			#range__ = np.arange(radar_i.nbins) * radar_i.range_res
			current_single = (int(100/len(self.com_radar)))*num
			data_ = create_cbb_elev_composite(self, radar_i, len(self.com_radar), current_single)
			data_ = convert_ppi_composite(radar_i, data_)
			data_ = data_.ravel()
			#data_com = np.where((data_com<data_) & (data_>=np.min(radar_i.elevs)), data_, data_com)
			data_ = np.nan_to_num(data_, nan=-9999)
			data_com = np.where(data_==-9999, data_com, data_)
			del data_
			gc.collect()
		
		self.signal.emit(100)
		data_com = np.where(data_com==-9999, np.nan, data_com)
		data_com = data_com.reshape(1980,1980)
		#save_radar_img(data_, gr="com", cm="ele", link_img=self.link_img)
		self.signal.emit(data_com)
		del self.com_radar, self.link_img
		del num, radar_i
		del current_single
		del data_com
		gc.collect()
		try:
			self.stop()
			gc.collect()
		except:
			pass

	def stop(self):
		#self.exit(0)
		self.exec()
		self.wait()

class CompositeThread_Height(QThread):
	signal = pyqtSignal('PyQt_PyObject')
	def __init__(self, com_radar, link_img):
		QThread.__init__(self)
		self.com_radar = com_radar
		self.link_img = link_img

	def run(self):
		data_com = np.zeros((1980*1980))
		data_com = np.where(data_com==0, -9999, -9999)
		for num, radar_i in enumerate(self.com_radar):
			range__ = np.arange(radar_i.nbins) * radar_i.range_res
			current_single = (int(100/len(self.com_radar)))*num
			data_ = create_cbb_hei_composite(self, radar_i, len(self.com_radar), current_single)
			data_ = convert_height(data_, range__, radar_i.site_coord[2])
			data_ = convert_ppi_composite(radar_i, data_)
			data_ = data_.ravel()
			#data_com = np.where((data_com<data_) & (data_>=np.min(radar_i.elevs)), data_, data_com)
			#data_com = np.where(data_com==-9999, data_, data_com)
			data_ = np.nan_to_num(data_, nan=-9999)
			data_com = np.where(data_==-9999, data_com, data_)
			del data_, range__
			gc.collect()
		self.signal.emit(100)
		data_com = np.where(data_com==-9999, np.nan, data_com)
		data_com = data_com.reshape(1980,1980)
		#save_radar_img(data_, gr="com", cm="hei", link_img=self.link_img)
		self.signal.emit(data_com)
		del self.com_radar, self.link_img
		del num, radar_i
		del current_single
		del data_com
		gc.collect()
		try:
			self.stop()
			gc.collect()
		except:
			pass

	def stop(self):
		#self.exit(0)
		self.exec()
		self.wait()

def create_cbb_elev_single_for_csv(radar_obj, el)->beam_block_doc:
	ds = wrl.io.open_raster("gt30e100n40.tif")
	
	rastervalues, rastercoords, proj = wrl.georef.extract_raster_dataset(ds, nodata=-32768.)
	data_ = np.zeros((radar_obj.nrays, radar_obj.nbins))
	data_ = np.where(data_==0, 9999, 9999)
	num=len(radar_obj.elevs)-1-el

	r = np.arange(radar_obj.nbins) * radar_obj.range_res
	beamradius = wrl.util.half_power_radius(r, radar_obj.beam_width)
	coord = wrl.georef.sweep_centroids(radar_obj.nrays, radar_obj.range_res, radar_obj.nbins, radar_obj.elevs[el])
	coords = wrl.georef.spherical_to_proj(coord[..., 0],
										  coord[..., 1],
										  coord[..., 2], radar_obj.site_coord)
	lon = coords[..., 0]
	lat = coords[..., 1]
	alt = coords[..., 2]
	polcoords = coords[..., :2]
	rlimits = (lon.min(), lat.min(), lon.max(), lat.max())
	
	ind = wrl.util.find_bbox_indices(rastercoords, rlimits)
	rastercoord = rastercoords[ind[1]:ind[3], ind[0]:ind[2], ...]
	rastervalue = rastervalues[ind[1]:ind[3], ind[0]:ind[2]]
	polarvalues = wrl.ipol.cart_to_irregular_spline(rastercoord, rastervalue,
												 polcoords, order=3,
												 prefilter=False)
	PBB = wrl.qual.beam_block_frac(polarvalues, alt, beamradius)
	PBB = np.ma.masked_invalid(PBB)
	CBB = wrl.qual.cum_beam_block_frac(PBB)
	data_ = np.where(CBB<0.2, radar_obj.elevs[el], data_)

	del radar_obj
	del ds, rastervalues, rastercoords, proj
	del el, r, beamradius, coord, coords
	del lon, lat, alt, polcoords, rlimits, ind
	del rastercoord, rastervalue
	del PBB, CBB
	gc.collect()
	return data_, polarvalues

class Create_CSV_Thread(QThread):
	signal = pyqtSignal('PyQt_PyObject')
	def __init__(self, radar_obj, file_csv):
		QThread.__init__(self)
		self.radar_obj = radar_obj
		self.file_csv = file_csv

	def run(self):
		self.signal.emit(0)
		range__ = np.arange(self.radar_obj.nbins) * self.radar_obj.range_res
		radar_elvs = self.radar_obj.elevs
		line_csv = [[] for i in range(self.radar_obj.nrays)]
		check_bin = np.zeros((self.radar_obj.nrays))
		cap_beam = 2000.
		num=0
		try:
			for i in range(len(radar_elvs)-1, -1, -1):
				self.signal.emit(int(int(100/len(radar_elvs))/5)+int(100/len(radar_elvs))*num)
				data_, polarvalues = create_cbb_elev_single_for_csv(self.radar_obj, i)
				heigt_ = convert_height(data_, range__, self.radar_obj.site_coord[2])
				height_ = np.where(heigt_==-9999, np.nan, heigt_-cap_beam)
				
				for j in range(360):
					if (j==200):
						self.signal.emit(int(int(100/len(radar_elvs))/3)+int(100/len(radar_elvs))*num)

					height_i = len(height_[j][np.where(height_[j]<=0)])+1 #num nbins
					range_km = range__[height_i]/1000.

					if (i==len(radar_elvs)-1):
						line_csv[j] = [j, 0]

					if (height_i>check_bin[j]):
						line_csv[j].extend([radar_elvs[i], range_km])
						check_bin[j] = height_i
					self.signal.emit(int(int(100/len(radar_elvs))/2)+int(100/len(radar_elvs))*num)
				num+=1
				self.signal.emit(int(int(100/len(radar_elvs))/1.1)+int(100/len(radar_elvs))*num)

			for i in range(360):
				line_csv[i][-1] = range__[-1]/1000.
			csv_file = open(self.file_csv,'w+',encoding="utf-8",newline='')
			file_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
			for line in line_csv:
				file_writer.writerow(line)
			csv_file.close()
			self.signal.emit(100)
			self.signal.emit("OK")
		except:
			self.signal.emit("Error")

		del range__, line_csv, check_bin, cap_beam
		del heigt_, height_, range_km, height_i
		del data_, polarvalues, self.radar_obj, self.file_csv
		del csv_file, file_writer
		del i, j, radar_elvs
		gc.collect()
		try:
			self.stop()
			gc.collect()
		except:
			pass


	def run_old(self):
		data_ = create_cbb_elev_single(self, self.radar_obj)
		range__ = np.arange(self.radar_obj.nbins) * self.radar_obj.range_res
		height_ = convert_height(data_, range__, self.radar_obj.site_coord[2])
		radar_elvs = self.radar_obj.elevs

		try:
			csv_file = open(self.file_csv,'w+',encoding="utf-8",newline='')
			file_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
			file_label = ["Ray"]
			for i in range(len(self.radar_obj.elevs)):
				file_label.extend([f"Elev_{i+1}", "From Bin", "To Bin"])
			file_writer.writerow(file_label)

			for i in range(data_.shape[0]):
				radar_elvs = self.radar_obj.elevs
				data_row = []
				data_row.append(i)
				for j in range(data_.shape[1]):
					for k, ele in enumerate(radar_elvs):
						if (data_[i,j]==ele):
							radar_elvs = np.delete(radar_elvs, k)
							if (len(data_row)==1):
								data_row.extend([data_[i,j], j])
							else:
								data_row.extend([j-1, data_[i,j], j])
				data_row.append(data_.shape[1])
				file_writer.writerow(data_row)
			csv_file.close()
			self.signal.emit(100)
			self.signal.emit("OK")
		except:
			self.signal.emit("Error")

		del data_, self.radar_obj, self.file_csv
		del csv_file, file_writer, file_label
		del i, j, k, ele, radar_elvs, data_row
		gc.collect()
		try:
			self.stop()
			gc.collect()
		except:
			pass

	def stop(self):
		#self.exit(0)
		self.exec()
		self.wait()

class MainWindow(QtWidgets.QMainWindow):
	def __init__(self):
		super().__init__()

	def closeEvent(self, event):
		close = QtWidgets.QMessageBox.question(self,
									 "Exit",
									 "Are you sure want to exit?",
									 QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
		if close == QtWidgets.QMessageBox.Yes:
			sys.exit()
		else:
			pass
			#event.ignore()


class Ui_MainWindow(object):
	global text_terminal
	text_terminal = "Terminal"
	def setupUi(self, MainWindow):
		MainWindow.setObjectName("MainWindow")
		#MainWindow.resize(725, 412)
		MainWindow.setFixedSize(725, 412)
		self.centralwidget = QtWidgets.QWidget(MainWindow)
		self.centralwidget.setObjectName("centralwidget")
		self.Progress_Bar = QtWidgets.QProgressBar(self.centralwidget)
		self.Progress_Bar.setGeometry(QtCore.QRect(80, 350, 391, 31))
		self.Progress_Bar.setProperty("value", 0)
		self.Progress_Bar.setObjectName("Progress_Bar")
		self.toolBox = QtWidgets.QToolBox(self.centralwidget)
		self.toolBox.setGeometry(QtCore.QRect(490, 90, 221, 191))
		self.toolBox.setLayoutDirection(QtCore.Qt.LeftToRight)
		self.toolBox.setAutoFillBackground(False)
		self.toolBox.setObjectName("toolBox")

		self.Label_AMO = QtWidgets.QLabel(self.centralwidget)
		self.Label_AMO.setGeometry(QtCore.QRect(600, 390, 121, 21))
		self.Label_AMO.setObjectName("Label_amo")

		self.page_1 = QtWidgets.QWidget()
		self.page_1.setGeometry(QtCore.QRect(0, 0, 221, 133))
		self.page_1.setObjectName("page_1")

		self.Label_Save_Radar = QtWidgets.QLabel(self.page_1)
		self.Label_Save_Radar.setGeometry(QtCore.QRect(10, 0, 111, 21))
		self.Label_Save_Radar.setObjectName("Label_Save_Radar")
		self.Label_plotEle = QtWidgets.QLabel(self.page_1)
		self.Label_plotEle.setGeometry(QtCore.QRect(10, 30, 111, 21))
		self.Label_plotEle.setObjectName("Label_plotEle")
		self.Label_Create_Csv = QtWidgets.QLabel(self.page_1)
		self.Label_Create_Csv.setGeometry(QtCore.QRect(10, 90, 111, 21))
		self.Label_Create_Csv.setObjectName("Label_Create_Csv")
		self.Label_plotHeight = QtWidgets.QLabel(self.page_1)
		self.Label_plotHeight.setGeometry(QtCore.QRect(10, 60, 111, 21))
		self.Label_plotHeight.setObjectName("Label_plotHeight")

		self.Save_Radar_Button = QtWidgets.QPushButton(self.page_1)
		self.Save_Radar_Button.setGeometry(QtCore.QRect(130, 0, 75, 23))
		self.Save_Radar_Button.setObjectName("Save_Radar_Button")
		self.Single_Ele = QtWidgets.QPushButton(self.page_1)
		self.Single_Ele.setGeometry(QtCore.QRect(130, 30, 75, 23))
		self.Single_Ele.setObjectName("Single_Ele")
		self.Single_Height = QtWidgets.QPushButton(self.page_1)
		self.Single_Height.setGeometry(QtCore.QRect(130, 60, 75, 23))
		self.Single_Height.setObjectName("Single_Height")
		self.Create_CSV = QtWidgets.QPushButton(self.page_1)
		self.Create_CSV.setGeometry(QtCore.QRect(130, 90, 75, 23))
		self.Create_CSV.setObjectName("Create_CSV")	
		self.toolBox.addItem(self.page_1, "")

		self.page_2 = QtWidgets.QWidget()
		self.page_2.setGeometry(QtCore.QRect(0, 0, 221, 133))
		self.page_2.setObjectName("page_2")

		self.Label_Add = QtWidgets.QLabel(self.page_2)
		self.Label_Add.setGeometry(QtCore.QRect(10, 0, 71, 21))
		self.Label_Add.setObjectName("Label_Add")
		self.Label_Com_Ele = QtWidgets.QLabel(self.page_2)
		self.Label_Com_Ele.setGeometry(QtCore.QRect(10, 30, 101, 21))
		self.Label_Com_Ele.setObjectName("Label_Com_Ele")
		self.Label_Com_Height = QtWidgets.QLabel(self.page_2)
		self.Label_Com_Height.setGeometry(QtCore.QRect(10, 60, 101, 21))
		self.Label_Com_Height.setObjectName("Label_Com_Height")
		self.Label_Clear = QtWidgets.QLabel(self.page_2)
		self.Label_Clear.setGeometry(QtCore.QRect(10, 90, 101, 21))
		self.Label_Clear.setObjectName("Label_Clear")

		self.Add_Com = QtWidgets.QPushButton(self.page_2)
		self.Add_Com.setGeometry(QtCore.QRect(130, 0, 75, 23))
		self.Add_Com.setObjectName("Add_Com")
		self.Create_Com_Ele = QtWidgets.QPushButton(self.page_2)
		self.Create_Com_Ele.setGeometry(QtCore.QRect(130, 30, 75, 23))
		self.Create_Com_Ele.setObjectName("Create_Com_Ele")	
		self.Create_Com_Height = QtWidgets.QPushButton(self.page_2)
		self.Create_Com_Height.setGeometry(QtCore.QRect(130, 60, 75, 23))
		self.Create_Com_Height.setObjectName("Create_Com_Height")
		self.Clear_Com = QtWidgets.QPushButton(self.page_2)
		self.Clear_Com.setGeometry(QtCore.QRect(130, 90, 75, 23))
		self.Clear_Com.setObjectName("Clear_Com")
		self.toolBox.addItem(self.page_2, "")


		self.Tab_Control = QtWidgets.QTabWidget(self.centralwidget)
		self.Tab_Control.setGeometry(QtCore.QRect(30, 40, 441, 191))
		self.Tab_Control.setObjectName("Tab_Control")
		self.tab_1 = QtWidgets.QWidget()
		self.tab_1.setObjectName("tab_1")

		self.Label_Name = QtWidgets.QLabel(self.tab_1)
		self.Label_Name.setGeometry(QtCore.QRect(20, 10, 111, 21))
		self.Label_Name.setObjectName("Label_Name")
		self.Label_Total_Ele = QtWidgets.QLabel(self.tab_1)
		self.Label_Total_Ele.setGeometry(QtCore.QRect(20, 90, 111, 21))
		self.Label_Total_Ele.setObjectName("Label_Total_Ele")
		self.Label_Ele = QtWidgets.QLabel(self.tab_1)
		self.Label_Ele.setGeometry(QtCore.QRect(20, 130, 111, 21))
		self.Label_Ele.setObjectName("Label_Ele")
		self.Label_Site = QtWidgets.QLabel(self.tab_1)
		self.Label_Site.setGeometry(QtCore.QRect(20, 50, 111, 21))
		self.Label_Site.setObjectName("Label_Site")

		self.Radar_Name = QtWidgets.QLineEdit(self.tab_1)
		self.Radar_Name.setGeometry(QtCore.QRect(160, 10, 251, 20))
		self.Radar_Name.setAlignment(QtCore.Qt.AlignCenter)
		self.Radar_Name.setObjectName("Radar_Name")
		self.Site_Coord = QtWidgets.QLineEdit(self.tab_1)
		self.Site_Coord.setGeometry(QtCore.QRect(160, 50, 251, 20))
		self.Site_Coord.setAlignment(QtCore.Qt.AlignCenter)
		self.Site_Coord.setObjectName("Site_Coord")
		self.Total_Ele = QtWidgets.QLineEdit(self.tab_1)
		self.Total_Ele.setGeometry(QtCore.QRect(160, 90, 251, 20))
		self.Total_Ele.setAlignment(QtCore.Qt.AlignCenter)
		self.Total_Ele.setObjectName("Total_Ele")
		self.Elevations = QtWidgets.QLineEdit(self.tab_1)
		self.Elevations.setGeometry(QtCore.QRect(160, 130, 251, 20))
		self.Elevations.setAlignment(QtCore.Qt.AlignCenter)
		self.Elevations.setReadOnly(False)
		self.Elevations.setObjectName("Elevations")
		
		self.Tab_Control.addTab(self.tab_1, "")

		self.tab_2 = QtWidgets.QWidget()
		self.tab_2.setObjectName("tab_2")

		self.Label_nrays = QtWidgets.QLabel(self.tab_2)
		self.Label_nrays.setGeometry(QtCore.QRect(20, 10, 111, 21))
		self.Label_nrays.setObjectName("Label_nrays")
		self.Label_nbins = QtWidgets.QLabel(self.tab_2)
		self.Label_nbins.setGeometry(QtCore.QRect(20, 50, 111, 21))
		self.Label_nbins.setObjectName("Label_nbins")
		self.Label_RangeRes = QtWidgets.QLabel(self.tab_2)
		self.Label_RangeRes.setGeometry(QtCore.QRect(20, 90, 131, 21))
		self.Label_RangeRes.setObjectName("Label_RangeRes")
		self.Label_BeamWidth = QtWidgets.QLabel(self.tab_2)
		self.Label_BeamWidth.setGeometry(QtCore.QRect(20, 130, 111, 21))
		self.Label_BeamWidth.setObjectName("Label_BeamWidth")

		self.nrays = QtWidgets.QLineEdit(self.tab_2)
		self.nrays.setGeometry(QtCore.QRect(160, 10, 251, 20))
		self.nrays.setAlignment(QtCore.Qt.AlignCenter)
		self.nrays.setReadOnly(False)
		self.nrays.setObjectName("nrays")
		self.nbins = QtWidgets.QLineEdit(self.tab_2)
		self.nbins.setGeometry(QtCore.QRect(160, 50, 251, 20))
		self.nbins.setAlignment(QtCore.Qt.AlignCenter)
		self.nbins.setReadOnly(False)
		self.nbins.setObjectName("nbins")
		self.range_res = QtWidgets.QLineEdit(self.tab_2)
		self.range_res.setGeometry(QtCore.QRect(160, 90, 251, 20))
		self.range_res.setAlignment(QtCore.Qt.AlignCenter)
		self.range_res.setReadOnly(False)
		self.range_res.setObjectName("range_res")
		self.beam_width = QtWidgets.QLineEdit(self.tab_2)
		self.beam_width.setGeometry(QtCore.QRect(160, 130, 251, 20))
		self.beam_width.setAlignment(QtCore.Qt.AlignCenter)
		self.beam_width.setObjectName("beam_width")

		self.Tab_Control.addTab(self.tab_2, "")
		self.tab_3 = QtWidgets.QWidget()
		self.tab_3.setObjectName("tab_3")

		self.Label_imgPath = QtWidgets.QLabel(self.tab_3)
		self.Label_imgPath.setGeometry(QtCore.QRect(20, 20, 111, 21))
		self.Label_imgPath.setObjectName("Label_imgPath")
		self.Label_RadarPath = QtWidgets.QLabel(self.tab_3)
		self.Label_RadarPath.setGeometry(QtCore.QRect(20, 50, 111, 21))
		self.Label_RadarPath.setObjectName("Label_RadarPath")
		self.Label_csvPath = QtWidgets.QLabel(self.tab_3)
		self.Label_csvPath.setGeometry(QtCore.QRect(20, 80, 111, 21))
		self.Label_csvPath.setObjectName("Label_csvPath")

		self.Img_Path = QtWidgets.QLineEdit(self.tab_3)
		self.Img_Path.setGeometry(QtCore.QRect(130, 20, 251, 20))
		self.Img_Path.setAlignment(QtCore.Qt.AlignCenter)
		self.Img_Path.setReadOnly(False)
		self.Img_Path.setObjectName("Img_Path")
		self.Radar_Path = QtWidgets.QLineEdit(self.tab_3)
		self.Radar_Path.setGeometry(QtCore.QRect(130, 50, 251, 20))
		self.Radar_Path.setAlignment(QtCore.Qt.AlignCenter)
		self.Radar_Path.setReadOnly(False)
		self.Radar_Path.setObjectName("Radar_Path")
		self.CSV_Path = QtWidgets.QLineEdit(self.tab_3)
		self.CSV_Path.setGeometry(QtCore.QRect(130, 80, 251, 20))
		self.CSV_Path.setAlignment(QtCore.Qt.AlignCenter)
		self.CSV_Path.setReadOnly(False)
		self.CSV_Path.setObjectName("CSV_Path")

		self.Search_Img_Path = QtWidgets.QToolButton(self.tab_3)
		self.Search_Img_Path.setGeometry(QtCore.QRect(400, 20, 21, 21))
		self.Search_Img_Path.setObjectName("Search_Radar")

		self.Search_Radar_Path = QtWidgets.QToolButton(self.tab_3)
		self.Search_Radar_Path.setGeometry(QtCore.QRect(400, 50, 21, 21))
		self.Search_Radar_Path.setObjectName("Search_Radar")

		self.Search_CSV_Path = QtWidgets.QToolButton(self.tab_3)
		self.Search_CSV_Path.setGeometry(QtCore.QRect(400, 80, 21, 21))
		self.Search_CSV_Path.setObjectName("Search_Radar")

		self.Save_Path = QtWidgets.QPushButton(self.tab_3)
		self.Save_Path.setGeometry(QtCore.QRect(330, 120, 81, 31))
		self.Save_Path.setObjectName("Save_Path")
		
		self.Tab_Control.addTab(self.tab_3, "")

		self.tab_4 = QtWidgets.QWidget()
		self.tab_4.setObjectName("tab_4")
		self.Search_Radar = QtWidgets.QToolButton(self.tab_4)
		self.Search_Radar.setGeometry(QtCore.QRect(400, 40, 21, 21))
		self.Search_Radar.setObjectName("Search_Radar")
		self.Load_Radar_Text = QtWidgets.QLineEdit(self.tab_4)
		self.Load_Radar_Text.setGeometry(QtCore.QRect(20, 40, 361, 21))
		self.Load_Radar_Text.setText("")
		self.Load_Radar_Text.setAlignment(QtCore.Qt.AlignCenter)
		self.Load_Radar_Text.setReadOnly(False)
		self.Load_Radar_Text.setObjectName("Load_Radar_Text")
		self.Load_Button = QtWidgets.QPushButton(self.tab_4)
		self.Load_Button.setGeometry(QtCore.QRect(300, 90, 81, 31))
		self.Load_Button.setObjectName("Load_Button")
		self.Tab_Control.addTab(self.tab_4, "")

		self.Terminal = QtWidgets.QTextBrowser(self.centralwidget)
		self.Terminal.setGeometry(QtCore.QRect(30, 280, 441, 51))
		self.Terminal.setLayoutDirection(QtCore.Qt.LeftToRight)
		self.Terminal.setObjectName("Terminal")
		self.Exit_Button = QtWidgets.QPushButton(self.centralwidget)
		self.Exit_Button.setGeometry(QtCore.QRect(560, 320, 111, 51))
		self.Exit_Button.setObjectName("Exit_Button")
		self.Submit_Button = QtWidgets.QPushButton(self.centralwidget)
		self.Submit_Button.setGeometry(QtCore.QRect(540, 30, 111, 31))
		self.Submit_Button.setObjectName("Submit_Button")
		
		MainWindow.setCentralWidget(self.centralwidget)

		self.retranslateUi(MainWindow)
		self.toolBox.setCurrentIndex(0)
		self.toolBox.layout().setSpacing(8)
		self.Tab_Control.setCurrentIndex(0)
		QtCore.QMetaObject.connectSlotsByName(MainWindow)

	def retranslateUi(self, MainWindow):
		global _translate
		_translate = QtCore.QCoreApplication.translate
		MainWindow.setWindowTitle(_translate("MainWindow", "Beam Block Application"))
		self.Label_AMO.setText(_translate("MainWindow", "lamtrant61@gmail.com"))

		self.Single_Ele.setText(_translate("MainWindow", "Create"))
		self.Label_plotEle.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt;\">Elevation plot:</span></p></body></html>"))
		self.Label_Create_Csv.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt;\">Create CSV:</span></p></body></html>"))
		self.Create_CSV.setText(_translate("MainWindow", "Create"))
		self.Label_plotHeight.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt;\">Height plot:</span></p></body></html>"))
		self.Single_Height.setText(_translate("MainWindow", "Create"))
		self.Label_Save_Radar.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt;\">Save radar:</span></p></body></html>"))
		self.Save_Radar_Button.setText(_translate("MainWindow", "Save"))
		self.toolBox.setItemText(self.toolBox.indexOf(self.page_1), _translate("MainWindow", "Single radar"))
		self.Add_Com.setText(_translate("MainWindow", "Add"))
		self.Label_Add.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt;\">Add radar:</span></p></body></html>"))
		self.Label_Com_Ele.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt;\">Elevation plot:</span></p></body></html>"))
		self.Create_Com_Ele.setText(_translate("MainWindow", "Create"))
		self.Label_Clear.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt;\">Clear radar:</span></p></body></html>"))
		self.Clear_Com.setText(_translate("MainWindow", "Clear"))
		self.Label_Com_Height.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt;\">Height plot:</span></p></body></html>"))
		self.Create_Com_Height.setText(_translate("MainWindow", "Create"))
		self.toolBox.setItemText(self.toolBox.indexOf(self.page_2), _translate("MainWindow", "Composite"))
		self.Label_Name.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt;\">Radar name:</span></p></body></html>"))
		self.Label_Total_Ele.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt;\">Total elevation: </span></p></body></html>"))
		self.Label_Ele.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt;\">Elevations (deg):</span></p></body></html>"))
		self.Label_Site.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt;\">Site coordinate: </span></p></body></html>"))
		self.Elevations.setText(_translate("MainWindow", "1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0"))
		self.Elevations.setPlaceholderText(_translate("MainWindow", "1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0"))
		self.Site_Coord.setText(_translate("MainWindow", "Lon,Lat,Alt"))
		self.Site_Coord.setPlaceholderText(_translate("MainWindow", "Lon,Lat,Alt"))
		self.Total_Ele.setText(_translate("MainWindow", "8"))
		self.Total_Ele.setPlaceholderText(_translate("MainWindow", "8"))
		self.Radar_Name.setText(_translate("MainWindow", "Radar name"))
		self.Radar_Name.setPlaceholderText(_translate("MainWindow", "Radar name"))
		self.Tab_Control.setTabText(self.Tab_Control.indexOf(self.tab_1), _translate("MainWindow", "Radar"))
		self.Label_nrays.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt;\">Nrays:</span></p></body></html>"))
		self.Label_nbins.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt;\">Nbins:</span></p></body></html>"))
		self.Label_RangeRes.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt;\">Range resolution (m):</span></p></body></html>"))
		self.nrays.setText(_translate("MainWindow", "360"))
		self.nrays.setPlaceholderText(_translate("MainWindow", "360"))
		self.nbins.setText(_translate("MainWindow", "400"))
		self.nbins.setPlaceholderText(_translate("MainWindow", "400"))
		self.range_res.setText(_translate("MainWindow", "1000"))
		self.range_res.setPlaceholderText(_translate("MainWindow", "1000"))
		self.Label_BeamWidth.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt;\">Beam width (deg): </span></p></body></html>"))
		self.beam_width.setText(_translate("MainWindow", "1.0"))
		self.beam_width.setPlaceholderText(_translate("MainWindow", "1.0"))
		self.Tab_Control.setTabText(self.Tab_Control.indexOf(self.tab_2), _translate("MainWindow", "Attribute"))
		self.Label_imgPath.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt;\">Image path: </span></p></body></html>"))
		self.Label_csvPath.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt;\">CSV path: </span></p></body></html>"))
		self.Img_Path.setText(_translate("MainWindow", "img/"))
		self.Img_Path.setPlaceholderText(_translate("MainWindow", "img/"))
		self.CSV_Path.setText(_translate("MainWindow", "csv/"))
		self.CSV_Path.setPlaceholderText(_translate("MainWindow", "csv/"))
		self.Save_Path.setText(_translate("MainWindow", "Save"))
		self.Label_RadarPath.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt;\">Radar path: </span></p></body></html>"))
		self.Radar_Path.setText(_translate("MainWindow", "radar/"))
		self.Radar_Path.setPlaceholderText(_translate("MainWindow", "csv/"))
		self.Tab_Control.setTabText(self.Tab_Control.indexOf(self.tab_3), _translate("MainWindow", "Save path"))
		self.Search_Radar.setText(_translate("MainWindow", "..."))

		self.Search_Img_Path.setText(_translate("MainWindow", "..."))
		self.Search_Radar_Path.setText(_translate("MainWindow", "..."))
		self.Search_CSV_Path.setText(_translate("MainWindow", "..."))

		self.Load_Radar_Text.setPlaceholderText(_translate("MainWindow", "File name"))
		self.Load_Button.setText(_translate("MainWindow", "Load"))
		self.Tab_Control.setTabText(self.Tab_Control.indexOf(self.tab_4), _translate("MainWindow", "Load radar"))
		self.Exit_Button.setText(_translate("MainWindow", "Exit"))
		self.Submit_Button.setText(_translate("MainWindow", "Submit radar"))
		self.Terminal.setPlaceholderText(_translate("MainWindow", "Terminal"))

		self.load_path_ini()
		self.nrays.setEnabled(False)

		self.create_tkinter_window()
		self.composite = []
		self.load_path()
		self.Save_Path.clicked.connect(self.save_path)
		self.Submit_Button.clicked.connect(self.submit_radar)
		self.Search_Radar.clicked.connect(self.search_radar)

		self.Search_Img_Path.clicked.connect(self.search_img_path)
		self.Search_Radar_Path.clicked.connect(self.search_radar_path)
		self.Search_CSV_Path.clicked.connect(self.search_csv_path)

		self.Save_Radar_Button.clicked.connect(self.save_radar)
		self.Load_Button.clicked.connect(self.load_radar)

		self.Single_Ele.clicked.connect(self.single_ele)
		self.Single_Height.clicked.connect(self.single_hei)
		self.Create_CSV.clicked.connect(self.create_csv)

		self.Create_Com_Ele.clicked.connect(self.com_ele)
		self.Create_Com_Height.clicked.connect(self.com_hei)
		self.Add_Com.clicked.connect(self.add_com)
		self.Clear_Com.clicked.connect(self.clear_composite)
		self.Exit_Button.clicked.connect(MainWindow.closeEvent)

	def load_path_ini(self):
		try:
			f = open("path.ini", "r+")
			img_path = self.decode_binary_string(f.readline().strip())
			radar_path = self.decode_binary_string(f.readline().strip())
			csv_path = self.decode_binary_string(f.readline().strip())
			f.close()

		except:
			current_path = str(pathlib.Path(__file__).parent.resolve()).strip().replace("\\","/") + "/"
			img_path=current_path+"img/"
			radar_path=current_path+"radar/"
			csv_path=current_path+"csv/"

		self.Radar_Path.setText(_translate("MainWindow", radar_path))
		self.Img_Path.setText(_translate("MainWindow", img_path))
		self.CSV_Path.setText(_translate("MainWindow", csv_path))


	def load_path(self):
		global text_terminal
		self.save_img_path = self.Img_Path.text()
		self.save_CSV_Path = self.CSV_Path.text()
		self.save_Radar_Path = self.Radar_Path.text()

		try:
			check_path(self.save_img_path)
			check_path(self.save_CSV_Path)
			check_path(self.save_Radar_Path)
		except:
			text_terminal="Path is not exist"
		self.Terminal.setText(_translate("MainWindow", text_terminal))

	def load_radar(self):
		global text_terminal
		text_terminal="Load radar"
		self.Terminal.setText(_translate("MainWindow", text_terminal))
		if os.path.exists(self.Load_Radar_Text.text()):
			try:
				file = open(self.Load_Radar_Text.text(), "r+")
				radar_param = []
				for line in file:
					param = line.replace("\n","")
					radar_param.append(param)
				file.close()
				radar_name__ = radar_param[0]
				radar_coord__ = radar_param[1]
				total_ele__ = radar_param[2]
				elevs__ = radar_param[3]
				nrays__ = radar_param[4]
				nbins__ = radar_param[5]
				range_res__ = radar_param[6]
				beam_width__ = radar_param[7]

				self.Radar_Name.setText(_translate("MainWindow", radar_name__))
				self.Site_Coord.setText(_translate("MainWindow", radar_coord__))
				self.Total_Ele.setText(_translate("MainWindow", total_ele__))
				self.Elevations.setText(_translate("MainWindow", elevs__))
				self.nrays.setText(_translate("MainWindow", nrays__))
				self.nbins.setText(_translate("MainWindow", nbins__))
				self.range_res.setText(_translate("MainWindow", range_res__))
				self.beam_width.setText(_translate("MainWindow", beam_width__))
				text_terminal="Success"
				self.Terminal.setText(_translate("MainWindow", text_terminal))

			except:
				text_terminal="Unknown file format"
				self.Terminal.setText(_translate("MainWindow", text_terminal))

		else:
			text_terminal="File is not exist"
			self.Terminal.setText(_translate("MainWindow", text_terminal))

	def save_path(self):
		global text_terminal
		self.save_img_path = self.Img_Path.text()
		self.save_CSV_Path = self.CSV_Path.text()
		self.save_Radar_Path = self.Radar_Path.text()
		try:
			check_path(self.save_img_path)
			check_path(self.save_CSV_Path)
			check_path(self.save_Radar_Path)
			f = open("path.ini", "w+")

			bi_img =  ''.join(format(ord(i), '08b') for i in self.save_img_path)
			f.write(bi_img)
			f.write("\n")

			bi_radar =  ''.join(format(ord(i), '08b') for i in self.save_Radar_Path)
			f.write(bi_radar)
			f.write("\n")

			bi_csv =  ''.join(format(ord(i), '08b') for i in self.save_CSV_Path)
			f.write(bi_csv)
			f.write("\n")

			f.close()

			text_terminal="Save path"
		except:
			text_terminal="Path is not exist"
		
		self.Terminal.setText(_translate("MainWindow", text_terminal))

	def search_radar(self):
		global text_terminal
		try:
			fname = QFileDialog.getOpenFileNames(directory=self.save_Radar_Path, filter="*.radar")
			self.Load_Radar_Text.setText(_translate("MainWindow", fname[0][0]))
			text_terminal = fname[0][0]
			self.Terminal.setText(_translate("MainWindow", text_terminal))

		except:
			if not os.path.exists(self.Load_Radar_Text.text()):
				text_terminal = "Invalid radar file"
				self.Terminal.setText(_translate("MainWindow", text_terminal))

	def decode_binary_string(self, string_decode):
		result = ""
		for i in range(int(len(string_decode)/8)):
			letter = chr(int(string_decode[i*8:i*8+8], 2))
			result+=letter
		return result

	def search_img_path(self):
		global text_terminal
		try:
			fname_img = QFileDialog.getExistingDirectory(directory=self.save_img_path) + "/"
			self.Img_Path.setText(_translate("MainWindow", fname_img))
		except:
			pass

	def search_radar_path(self):
		global text_terminal
		try:
			fname_radar = QFileDialog.getExistingDirectory(directory=self.save_Radar_Path) + "/"
			self.Radar_Path.setText(_translate("MainWindow", fname_radar))
		except:
			pass

	def search_csv_path(self):
		global text_terminal
		try:
			fname_csv = QFileDialog.getExistingDirectory(directory=self.save_CSV_Path) + "/"
			self.CSV_Path.setText(_translate("MainWindow", fname_csv))
		except:
			pass

	def save_radar(self):
		global text_terminal
		text_terminal="Save radar"
		self.Terminal.setText(_translate("MainWindow", text_terminal))

		try:
			save_name = self.save_Radar_Path + str(self.single_radar.radar_name) + "_" + str(self.single_radar.site_coord[0]) + "_" + str(self.single_radar.site_coord[1]) + "_" + str(self.single_radar.site_coord[2]) + "_" + str(len(self.single_radar.elevs)) + "_" + str(self.single_radar.beam_width) + ".radar"
			file = open(save_name, "w+")
			file.write(str(self.single_radar.radar_name))
			file.write("\n")
			file.write(str(self.single_radar.site_coord[0]) + "," + str(self.single_radar.site_coord[1]) + "," + str(self.single_radar.site_coord[2]))
			file.write("\n")
			file.write(str(len(self.single_radar.elevs)))
			file.write("\n")
			for i, el in enumerate(self.single_radar.elevs):
				if (i==len(self.single_radar.elevs)-1):
					file.write(str(el))
					file.write("\n")
				else:
					file.write(str(el))
					file.write(",")
			file.write(str(self.single_radar.nrays))
			file.write("\n")
			file.write(str(self.single_radar.nbins))
			file.write("\n")
			file.write(str(self.single_radar.range_res))
			file.write("\n")
			file.write(str(self.single_radar.beam_width))
			file.close()
			text_terminal += "\nSuccess"
			self.Terminal.setText(_translate("MainWindow", text_terminal))

		except:
			text_terminal += "\nSome error occur"
			self.Terminal.setText(_translate("MainWindow", text_terminal))

	def check_vn(self, lon, lat):
		check__ = True
		if not (100<=lon<=110):
			check__ = False
		if not (8<=lat<=24):
			check__ = False
		return check__

	def submit_radar(self):
		global text_terminal
		check_submit = True
		text_terminal = "Submit radar"
		radar_name = self.Radar_Name.text()

		if (radar_name=="Radar name"):
			text_terminal+="\nInvalid Radar Name"
			check_submit = False

		radar_coor = self.Site_Coord.text()
		try:
			radar_coor = np.array(radar_coor.split(",")).astype("float64")
			if (len(radar_coor)==3):
				lon = radar_coor[0]
				lat = radar_coor[1]
				alt = radar_coor[2]
			else:
				text_terminal+="\nRadar coord must = 3"
				check_submit = False
			check__ = self.check_vn(lon, lat)
			if not check__:
				text_terminal+="\nRadar must be in range \nLon: 100~110, Lat:8~24"
				check_submit = False
		except:
			text_terminal+="\nInvalid Radar Coord"
			check_submit = False


		try:
			total_ele = int(self.Total_Ele.text())
		except:
			text_terminal+="\nTotal ele must be integer"
			check_submit = False
		elevs = self.Elevations.text()


		try:
			elevs = np.array(elevs.split(",")).astype("float64")
			if (len(elevs)!=total_ele):
				text_terminal+="\nTotal elevs must be the same as elevations"
				check_submit = False
		except:
			text_terminal+="\nInvalid elevations"
			check_submit = False


		try:
			nrays = int(self.nrays.text())
		except:
			text_terminal+="\nNrays must be integer"
			check_submit = False


		try:
			nbins = int(self.nbins.text())
		except:
			text_terminal+="\nNbins must be integer"
			check_submit = False


		try:
			range_res = float(self.range_res.text())
		except:
			text_terminal+="\nRange resolution must be float"
			check_submit = False


		try:
			beam_width = float(self.beam_width.text())
		except:
			text_terminal+="\nBeam width must be float"
			check_submit = False


		if check_submit:
			self.single_radar = create_radar(radar_name, lon, lat, alt, nrays, nbins, elevs, range_res, beam_width)
			text_terminal+="\nSuccess\n"
		self.Terminal.setText(_translate("MainWindow", text_terminal))


		if check_submit:
			self.display_radar()

	def check_submit_radar(self):
		if hasattr(self, 'single_radar'):
			return True
		else:
			return False
			
	def display_radar(self):
		global text_terminal
		text_terminal += str(self.single_radar.__dict__).replace("{","").replace("}","")
		self.Terminal.setText(_translate("MainWindow", text_terminal))


	def create_tkinter_window(self):
		self.root = Tk.Tk()
		self.root.wm_title("Radar plot")
		self.root.resizable(0,0)

		toolbar_frame = Tk.Frame(self.root)
		toolbar_frame.pack(side="top", fill="both", expand=0)

		fig = plt.figure(figsize=(6.5,6.5))
		canvas = FigureCanvasTkAgg(fig, master=self.root)
		#canvas.draw()
		canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
		toolbar = NavigationToolbar2TkAgg(canvas, toolbar_frame) 
		#toolbar.update() 

	def clear_plot(self):
		global im	
		try:
			self.root.destroy()
			plt.clf()
			plt.close()
		except:
			pass
		gc.collect()

	def single_ele_plot(self, data):
		global im
		icon_photo = PhotoImage(file='icon/radar_icon.png')
		self.root.iconphoto(False, icon_photo)
		cmap,norm,bounds = ELEVS_CMAP()
		#bounds = bounds-0.001
		map = Basemap(llcrnrlon=(self.single_radar.com_grid["lonmin"]),llcrnrlat=(self.single_radar.com_grid["latmin"]),urcrnrlon=(self.single_radar.com_grid["lonmax"]),urcrnrlat=(self.single_radar.com_grid["latmax"]), epsg=4326, ellps='WGS84')
		im = map.imshow(data, origin="lower",cmap=cmap,norm=norm)
		map.readshapefile('shapefile/sss','sss', linewidth=0.5, color='k',  default_encoding='utf-8')
		map.drawparallels(np.arange(-90.0, 90.0, 2), linewidth=0.4, dashes=[4, 4], color='blue', labels=[True,False,False,True], fmt='%g',size=6)
		map.drawmeridians(np.arange(0.0, 360.0, 2), linewidth=0.4, dashes=[4, 4], color='blue', labels=[True,False,False,True], fmt='%g',size=6)
		plt.text(112.8, 9.5, "Q. Trường Sa", ha="center", family='sans-serif', size=8, color="brown")
		plt.text(111.5, 17.0, "Q. Hoàng Sa", ha="center", family='sans-serif', size=8, color="brown")
		cbar = plt.colorbar(cmap=cmap, ticks=bounds, orientation='vertical', 
					shrink=2.4, aspect=36, fraction=0.02, pad=0.08,
					extend='both', extendfrac=None, extendrect=False)
		cbar.set_label("Elevs.[°]", labelpad=-20, y=1.05, rotation=0)
		plt.xlabel('Longitude [deg]', labelpad=20.0)
		plt.ylabel('Latitude [deg]', labelpad=32.0)
		plt.title('Single radar elevations')
		del data
		gc.collect()
		Tk.mainloop()

	def com_ele_plot(self, data):
		global im
		icon_photo = PhotoImage(file='icon/radar_icon.png')
		self.root.iconphoto(False, icon_photo)
		cmap,norm,bounds = ELEVS_CMAP()
		#bounds = bounds-0.001
		map = Basemap(llcrnrlon=97,llcrnrlat=7.2,urcrnrlon=115,urcrnrlat=25.2, epsg=4326, ellps='WGS84')
		im = map.imshow(data, origin="lower",cmap=cmap,norm=norm)
		map.readshapefile('shapefile/sss','sss', linewidth=0.5, color='k',  default_encoding='utf-8')
		map.drawparallels(np.arange(-90.0, 90.0, 2), linewidth=0.4, dashes=[4, 4], color='blue', labels=[True,False,False,True], fmt='%g',size=6)
		map.drawmeridians(np.arange(0.0, 360.0, 2), linewidth=0.4, dashes=[4, 4], color='blue', labels=[True,False,False,True], fmt='%g',size=6)
		plt.text(112.8, 9.5, "Q. Trường Sa", ha="center", family='sans-serif', size=8, color="brown")
		plt.text(111.5, 17.0, "Q. Hoàng Sa", ha="center", family='sans-serif', size=8, color="brown")
		cbar = plt.colorbar(cmap=cmap, ticks=bounds, orientation='vertical', 
					shrink=2.4, aspect=36, fraction=0.02, pad=0.08,
					extend='both', extendfrac=None, extendrect=False)
		cbar.set_label("Elevs.[°]", labelpad=-20, y=1.05, rotation=0)
		plt.xlabel('Longitude [deg]', labelpad=20.0)
		plt.ylabel('Latitude [deg]', labelpad=32.0)
		plt.title('Composite radar elevations')
		del data
		gc.collect()
		Tk.mainloop()

	def com_hei_plot(self, data):
		global im
		icon_photo = PhotoImage(file='icon/radar_icon.png')
		self.root.iconphoto(False, icon_photo)
		cmap,norm,bounds = HEIGHT_CMAP()
		#bounds = bounds-0.001
		map = Basemap(llcrnrlon=97,llcrnrlat=7.2,urcrnrlon=115,urcrnrlat=25.2, epsg=4326, ellps='WGS84')
		im = map.imshow(data/1000, origin="lower",cmap=cmap,norm=norm)
		map.readshapefile('shapefile/sss','sss', linewidth=0.5, color='k',  default_encoding='utf-8')
		map.drawparallels(np.arange(-90.0, 90.0, 2), linewidth=0.4, dashes=[4, 4], color='blue', labels=[True,False,False,True], fmt='%g',size=6)
		map.drawmeridians(np.arange(0.0, 360.0, 2), linewidth=0.4, dashes=[4, 4], color='blue', labels=[True,False,False,True], fmt='%g',size=6)
		plt.text(112.8, 9.5, "Q. Trường Sa", ha="center", family='sans-serif', size=8, color="brown")
		plt.text(111.5, 17.0, "Q. Hoàng Sa", ha="center", family='sans-serif', size=8, color="brown")
		cbar = plt.colorbar(cmap=cmap, ticks=bounds, orientation='vertical', 
					shrink=2.4, aspect=36, fraction=0.02, pad=0.08,
					extend='both', extendfrac=None, extendrect=False)
		cbar.set_label("H.[km]", labelpad=-20, y=1.05, rotation=0)
		plt.xlabel('Longitude [deg]', labelpad=20.0)
		plt.ylabel('Latitude [deg]', labelpad=32.0)
		plt.title('Composite radar height')
		del data
		gc.collect()
		Tk.mainloop()

	def single_height_plot(self, data):
		global im
		icon_photo = PhotoImage(file='icon/radar_icon.png')
		self.root.iconphoto(False, icon_photo)
		cmap,norm,bounds = HEIGHT_CMAP()
		map = Basemap(llcrnrlon=(self.single_radar.com_grid["lonmin"]),llcrnrlat=(self.single_radar.com_grid["latmin"]),urcrnrlon=(self.single_radar.com_grid["lonmax"]),urcrnrlat=(self.single_radar.com_grid["latmax"]), epsg=4326, ellps='WGS84')
		im = map.imshow(data/1000., origin="lower",cmap=cmap,norm=norm)
		map.readshapefile('shapefile/sss','sss', linewidth=0.5, color='k',  default_encoding='utf-8')
		map.drawparallels(np.arange(-90.0, 90.0, 2), linewidth=0.4, dashes=[4, 4], color='blue', labels=[True,False,False,True], fmt='%g',size=6)
		map.drawmeridians(np.arange(0.0, 360.0, 2), linewidth=0.4, dashes=[4, 4], color='blue', labels=[True,False,False,True], fmt='%g',size=6)
		plt.text(112.8, 9.5, "Q. Trường Sa", ha="center", family='sans-serif', size=8, color="brown")
		plt.text(111.5, 17.0, "Q. Hoàng Sa", ha="center", family='sans-serif', size=8, color="brown")
		cbar = plt.colorbar(cmap=cmap, ticks=bounds, orientation='vertical', 
					shrink=2.4, aspect=36, fraction=0.02, pad=0.08,
					extend='both', extendfrac=None, extendrect=False)
		cbar.set_label("H.[km]", labelpad=-20, y=1.05, rotation=0)
		plt.xlabel('Longitude [deg]', labelpad=20.0)
		plt.ylabel('Latitude [deg]', labelpad=32.0)
		plt.title('Single radar height')
		del data
		gc.collect()
		Tk.mainloop()

	def single_ele(self):
		global text_terminal
		text_terminal="Single elevation"
		self.Terminal.setText(_translate("MainWindow", text_terminal))
		if not self.check_submit_radar():
			text_terminal+="\nYou must submit radar before plot"
			self.Terminal.setText(_translate("MainWindow", text_terminal))
		else:
			text_terminal+="\nRadar " + self.single_radar.radar_name
			self.Terminal.setText(_translate("MainWindow", text_terminal))
			try:
				del self.single_thread
				self.clear_plot()
				#print ("OK")
				gc.collect()
			except:
				pass
			self.single_thread = SingleThread_Elevs(self.single_radar, self.save_img_path)
			self.single_thread.signal.connect(self.create_ele_single)
			self.block_all_button()
			self.single_thread.start()


	def single_hei(self):
		global text_terminal
		text_terminal="Single height"
		self.Terminal.setText(_translate("MainWindow", text_terminal))
		if not self.check_submit_radar():
			text_terminal+="\nYou must submit radar before plot"
			self.Terminal.setText(_translate("MainWindow", text_terminal))
		else:
			text_terminal+="\nRadar " + self.single_radar.radar_name
			self.Terminal.setText(_translate("MainWindow", text_terminal))
			try:
				del self.single_thread
				self.clear_plot()
				gc.collect()
			except:
				pass
			self.single_thread = SingleThread_Height(self.single_radar, self.save_img_path)
			self.single_thread.signal.connect(self.create_hei_single)
			self.block_all_button()
			self.single_thread.start()


	def add_com(self):
		global text_terminal
		text_terminal="Add radar to composite"
		self.Terminal.setText(_translate("MainWindow", text_terminal))
		try:
			check_add = True
			for radar_obj in self.composite:
				if (radar_obj==self.single_radar):
					check_add = False
					break
			if check_add:
				self.composite.append(self.single_radar)
				text_terminal += "  -  Radar " + str(self.single_radar.radar_name)
				text_terminal += "\nTotal radar " + str(len(self.composite))

				str_name_com_radar = ""
				for i in range(len(self.composite)):
					if (i==0):
						str_name_com_radar+=self.composite[i].radar_name
					else:
						str_name_com_radar+=", "+self.composite[i].radar_name

				text_terminal += "\n" + str_name_com_radar
				self.Terminal.setText(_translate("MainWindow", text_terminal))
			else:
				text_terminal += "\nRadar name or site coordinate has added before"
				self.Terminal.setText(_translate("MainWindow", text_terminal))

		except:
			text_terminal += "\nSome error occur"
			self.Terminal.setText(_translate("MainWindow", text_terminal))

	def com_ele(self):
		global text_terminal
		text_terminal="Composite elevation"
		self.Terminal.setText(_translate("MainWindow", text_terminal))
		if (len(self.composite)==0):
			text_terminal+="\nYou must add radar before composite"
			self.Terminal.setText(_translate("MainWindow", text_terminal))
		else:
			text_terminal+="\nRadar "
			for i in range(len(self.composite)):
				if (i==len(self.composite)-1):
					text_terminal+=self.composite[i].radar_name
				else:
					text_terminal+=self.composite[i].radar_name+", "
			self.Terminal.setText(_translate("MainWindow", text_terminal))

			try:
				del self.single_thread
				self.clear_plot()
				gc.collect()
			except:
				pass
			self.single_thread = CompositeThread_Elevs(self.composite, self.save_img_path)
			self.single_thread.signal.connect(self.create_ele_composite)
			self.block_all_button()
			self.single_thread.start()


	def com_hei(self):
		global text_terminal
		text_terminal="Composite height"
		self.Terminal.setText(_translate("MainWindow", text_terminal))
		if (len(self.composite)==0):
			text_terminal+="\nYou must add radar before composite"
			self.Terminal.setText(_translate("MainWindow", text_terminal))
		else:
			text_terminal+="\nRadar "
			for i in range(len(self.composite)):
				if (i==len(self.composite)-1):
					text_terminal+=self.composite[i].radar_name
				else:
					text_terminal+=self.composite[i].radar_name+", "
			self.Terminal.setText(_translate("MainWindow", text_terminal))
			try:
				del self.single_thread
				self.clear_plot()
				gc.collect()
			except:
				pass
			self.single_thread = CompositeThread_Height(self.composite, self.save_img_path)
			self.single_thread.signal.connect(self.create_hei_composite)
			self.block_all_button()
			self.single_thread.start()


	def clear_composite(self):
		global text_terminal
		text_terminal="Clear all radar"
		self.Terminal.setText(_translate("MainWindow", text_terminal))
		try:
			self.composite.clear()
		except:
			self.composite = []

	def create_csv(self):
		global text_terminal
		text_terminal="Create CSV"
		self.Terminal.setText(_translate("MainWindow", text_terminal))

		if not self.check_submit_radar():
			text_terminal+="\nYou must submit radar before create csv file"
			self.Terminal.setText(_translate("MainWindow", text_terminal))
		else:
			file_csv = self.save_CSV_Path + str(self.single_radar.radar_name) + "_" + str(self.single_radar.site_coord[0]) + "_" + str(self.single_radar.site_coord[1]) + "_" + str(self.single_radar.site_coord[2]) + "_" + str(len(self.single_radar.elevs)) + "_" + str(self.single_radar.beam_width) + ".csv"
			
			try:
				del self.single_thread
				self.clear_plot()
				gc.collect()
			except:
				pass

			self.single_thread = Create_CSV_Thread(self.single_radar, file_csv)
			self.single_thread.signal.connect(self.create_csv_data)
			self.block_all_button()
			self.single_thread.start()


	def block_all_button(self):
		self.Save_Path.setEnabled(False)
		self.Search_Radar.setEnabled(False)
		self.Search_Img_Path.setEnabled(False)
		self.Search_Radar_Path.setEnabled(False)
		self.Search_CSV_Path.setEnabled(False)
		self.Load_Button.setEnabled(False)

		self.Submit_Button.setEnabled(False)
		self.Save_Radar_Button.setEnabled(False)
		self.Single_Ele.setEnabled(False)
		self.Single_Height.setEnabled(False)
		self.Create_CSV.setEnabled(False)
		self.Create_Com_Ele.setEnabled(False)
		self.Create_Com_Height.setEnabled(False)
		self.Add_Com.setEnabled(False)
		self.Clear_Com.setEnabled(False)

	def unblock_all_button(self):
		self.Save_Path.setEnabled(True)
		self.Search_Radar.setEnabled(True)
		self.Search_Img_Path.setEnabled(True)
		self.Search_Radar_Path.setEnabled(True)
		self.Search_CSV_Path.setEnabled(True)
		self.Load_Button.setEnabled(True)

		self.Submit_Button.setEnabled(True)
		self.Save_Radar_Button.setEnabled(True)
		self.Single_Ele.setEnabled(True)
		self.Single_Height.setEnabled(True)
		self.Create_CSV.setEnabled(True)
		self.Create_Com_Ele.setEnabled(True)
		self.Create_Com_Height.setEnabled(True)
		self.Add_Com.setEnabled(True)
		self.Clear_Com.setEnabled(True)


	def create_ele_single(self, single_data):
		global text_terminal
		try:
			value_bar = int(single_data)
			self.Progress_Bar.setValue(value_bar)
		except:
			save_radar_img(single_data, radar_obj=self.single_radar, gr="SINGLE", cm="ELE", link_img=self.save_img_path)
			self.unblock_all_button()
			try:
				self.single_ele_plot(single_data)
			except:
				self.create_tkinter_window()
				self.single_ele_plot(single_data)
			try:
				del single_data
				gc.collect()
			except:
				pass

		

	def create_hei_single(self, single_data):
		global text_terminal
		try:
			value_bar = int(single_data)
			self.Progress_Bar.setValue(value_bar)
		except:
			save_radar_img(single_data, radar_obj=self.single_radar, gr="SINGLE", cm="HEI", link_img=self.save_img_path)
			self.unblock_all_button()
			try:
				self.single_height_plot(single_data)
			except:
				self.create_tkinter_window()
				self.single_height_plot(single_data)
			try:
				del single_data
				gc.collect()
			except:
				pass

	def create_ele_composite(self, com_data):
		global text_terminal
		try:
			value_bar = int(com_data)
			self.Progress_Bar.setValue(value_bar)
		except:
			str_com ="_"
			for radar_i in self.composite:
				str_com += radar_i.radar_name + "_"

			save_radar_img(com_data, gr="COM", cm="ELE", str_com= str_com, link_img=self.save_img_path)
			self.unblock_all_button()
			try:
				self.com_ele_plot(com_data)
			except:
				self.create_tkinter_window()
				self.com_ele_plot(com_data)
			try:
				del single_data
				gc.collect()
			except:
				pass

		
	def create_hei_composite(self, com_data):
		global text_terminal
		try:
			value_bar = int(com_data)
			self.Progress_Bar.setValue(value_bar)
		except:
			str_com = "_"
			for radar_i in self.composite:
				str_com += radar_i.radar_name + "_"

			save_radar_img(com_data, gr="COM", cm="HEI", str_com= str_com, link_img=self.save_img_path)
			self.unblock_all_button()
			try:
				self.com_hei_plot(com_data)
			except:
				self.create_tkinter_window()
				self.com_hei_plot(com_data)
			try:
				del single_data
				gc.collect()
			except:
				pass

			
	def create_csv_data(self, data_respond):
		global text_terminal
		try:
			value_bar = int(data_respond)
			self.Progress_Bar.setValue(value_bar)
		except:
			self.unblock_all_button()
			if (data_respond=="OK"):
				text_terminal+="\nSuccess"
				self.Terminal.setText(_translate("MainWindow", text_terminal))
			else:
				text_terminal+="\nSome error occur, may be you are opening file csv with the same name"
				self.Terminal.setText(_translate("MainWindow", text_terminal))
			try:
				del data_respond
				gc.collect()
			except:
				pass
	
	def exit_app(self):
		sys.exit()


stylesheet = """
	MainWindow {
		background-image: url("background.jpg"); 
		background-repeat: no-repeat; 
		background-position: center;
	}
"""

if __name__ == "__main__":
	gc.enable()
	app = QtWidgets.QApplication(sys.argv)
	MainWindow = MainWindow()
	MainWindow.setWindowIcon(QIcon("icon/radar_icon.png"))
	app.setStyleSheet(stylesheet)
	#MainWindow.setWindowFlag(QtCore.Qt.WindowCloseButtonHint, False)
	ui = Ui_MainWindow()
	ui.setupUi(MainWindow)
	MainWindow.show()
	sys.exit(app.exec_())

