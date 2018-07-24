function Data() {
    
    this.prefixes = ['uvOCRSLT', 'uvOCRSLTF'];
    this.prefix_names = ['Time-averaged', 'Time-avg. + delay filtered']
    this.pols = ['XX', 'YY', 'pI', 'pQ'];
    this.red_bls = [
        [ [11,12], [12,13] ], 
        [ [1,2], [2,3], [3,4] ], 
        [ [100,101], [101,102] ],
    ];
    this.lsts = [ 1.00641, 1.10036, 1.19432, 1.28828, 1.38224, 1.47620, 
                  1.57015, 1.66411, 1.75807, 1.85203, 1.94599 ];
};
