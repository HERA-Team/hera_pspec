import numpy as np
import os.path
import datetime
import astropy.time
import ephem

# FIXME: Use more precise latitude/longitude for HERA site
HERA_LAT = -30.739472
HERA_LONG = 21.442917

#-------------------------------------------------------------------------------
# Time-based grouping
#-------------------------------------------------------------------------------

def by_hourangle():
    """
    Group datafiles according to the hour angle of observation.
    """
    NotImplementedError()

def by_solar_elevation():
    """
    Group datafiles according to the elevation of the Sun at the time of 
    observation.
    """
    NotImplementedError()

def by_julian_date():
    """
    Group datafiles according to the Julian date of observation.
    """
    NotImplementedError()

def by_moon_position():
    """
    Group datafiles according to the position of the Moon with respect to the 
    center of the HERA field of view.
    """
    # Set observer position, date, and time
    obs = ephem.Observer()
    obs.lat, obs.lon = HERA_LAT, HERA_LONG
    obs.date = datetime.datetime(2017, 10, 8, 11, 23, 42) # FIXME
    
    print("Moon = %f, %f" % (ephem.Moon(obs).alt, ephem.Moon(obs).az))

def by_orbcomm_position():
    """
    Group datafiles according to the position of the Orbcomm RFI source with 
    respect to the center of the HERA field of view.
    """
    
    def get_orbcomm_tles(start_date=(17, 10), end_date=(18, 01), 
                         url="https://www.orbcomm.com/uploads/files/o%d.tle"):
        """
        Download Orbcomm two-line element (TLE) sets, used to calculate 
        satellite positions as a function of time.
        
        Parameters
        ----------
        start_date, end_date : tuple, optional
            Tuples of (year, day), where day is the number of days since the 
            start of the year. TLEs will be downloaded for all days between the 
            start date and end date, inclusive.
        
        url : str, optional
            Base URL to download the TLEs from.
        """
        import urllib2
        years = range(start_date[0], end_date[0]+1)
        
        #astropy.time.Time
        for i, yr in enumerate(years):
            
            # Set start and end days
            s_day, e_day = 1, 365 # FIXME: Leap years
            if i == 0: s_day = start_date[1]
            if i+1 == len(years): e_day = end_date[1]
            
            # Loop over days
            for day in range(s_day, e_day+1):
                # Try to download TLE file if necessary
                date_str = yr * 1000 + day
                fname = "data/orbcomm_tle_%d.txt" % date_str
                if os.path.isfile(fname): continue
                try:
                    print("\tDownloading Orbcomm TLE: %s" % date_str)
                    response = urllib2.urlopen(url % date_str, timeout=5)
                    tle_data = response.read()
                    f = open(fname, 'w')
                    f.write(tle_data)
                    f.close()
                except urllib2.URLError as e:
                    print(e)
    
    
    def load_tle_for_jd(jd):
        """
        Load TLE for a given Julian Date.
        """
        # FIXME
        yr = 17
        day = 213
        
        # Load data file
        date_str = yr * 1000 + day # FIXME
        try:
            fname = "data/orbcomm_tle_%d.txt" % date_str
            f = open(fname, 'r')
            lines = f.readlines()
            f.close()
        except:
            raise IOError("Unable to load file '%s'." % fname)
        
        names = []; line1 = []; line2 = []
        for line in lines:
            # Parse line
            if line == "\n": continue
            if "(" in line:
                names.append(line[:-1])
                continue
            if line[0] == '1': line1.append(line[:-1])
            if line[0] == '2': line2.append(line[:-1])
        
        # Sanity check on parsing
        if len(names) != len(line1) or len(names) != len(line2):
            raise ValueError("Failed to parse TLE file; mismatch in number "
                             "of entries that were found.")
        
        # Set observer position, date, and time
        obs = ephem.Observer()
        obs.lat, obs.lon = HERA_LAT, HERA_LONG
        obs.date = datetime.datetime(2017, 10, 8, 11, 23, 53) # FIXME
        
        # Load TLE in pyephem and convert to alt/az
        for i in [1,]: #range(len(names)):
            
            tle = ephem.readtle(names[i], line1[i], line2[i])
            tle.compute(obs)
            print(tle.alt, tle.az) # FIXME
    
    #get_orbcomm_tles()
    load_tle_for_jd(None)
    #NotImplementedError()

def by_ptsrc_position():
    """
    Group datafiles according to the position of bright point sources with 
    respect to the center of the HERA field of view.
    """
    NotImplementedError()



#-------------------------------------------------------------------------------
# Environmental grouping
#-------------------------------------------------------------------------------

def by_ambient_temp():
    """
    Group datafiles according to the ambient temperature during the observation.
    """
    NotImplementedError()

def by_rfi_flag_fraction():
    """
    Group datafiles according to the fraction of frequency channels and LSTs 
    that were flagged due to RFI.
    """
    NotImplementedError()

def by_rfi_peak_amplitude():
    """
    Group datafiles according to the peak detected RFI intensity during the 
    observation.
    """
    NotImplementedError()



#-------------------------------------------------------------------------------
# Hardware configuration-based grouping
#-------------------------------------------------------------------------------

def by_hex():
    """
    Group baselines according to the HERA hexes that their constituent antennas 
    come from.
    """
    NotImplementedError()

def by_antenna_year():
    """
    Group baselines according to the years that their constituent antennas were 
    built/deployed.
    """
    NotImplementedError()

def by_rack_position():
    """
    Group baselines according to the position of the inputs of their 
    constituent antennas in the correlator rack (to test for cross-talk).
    """
    NotImplementedError()


