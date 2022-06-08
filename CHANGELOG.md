v0.2.0 (2019-08-09)
    * Python 3 compatibility.
    * Allow cross-polarization spectra to be calculated as long as they
      aren't beam-normalized; polarizations now specified as polpairs.
    * Time-dependent noise power spectra capability in generate_noise.
    * Methods to fetch redundant baselines/blpairs from UVPSpec objects.
    * Exact normalization mode (Saurabh Singh) with optimization (Ronan
      Legin).
    * Updated covariance handling, incl. averaging and analytic variance
      (Jianrong Tan).
    * New 'lazy covariance' weighting mode (Aaron Ewall-Wice).
    * Fix bug where little_h was passed incorrectly (Zachary Martinot).
    * Store additional statistical info in stats_array (Duncan Rocha).
    * Add delay wedge plotting function (Paul Chichura).
    * Various minor interface changes and improved tests.

v0.1.0 (2018-07-18)
    * Initial release; implements core functionality,
      documentation, and tests.
    * Includes basic pipeline run scripts.
