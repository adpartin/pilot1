"""
Class CombinedRNASeqLINCS that loads RNA-Seq LINCS data.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


class CombinedRNASeqLINCS():
    """ Combined LINCS dataset. """
    def __init__(self, datadir, cellmeta_fname, rna_norm='raw', sources=[],
                 na_values=['na', '-', ''], verbose=True):
        """ Note that rna file must have the following structure:
        rna.columns[0] --> 'Sample'
        rna.columns[1:] --> gene names
        rna.iloc[:, 0] --> strings of sample names
        rna.iloc[:, 1:] --> gene expression values
        
        Example:
            DATADIR = '/Users/apartin/work/jdacs/Benchmarks/Data/Pilot1'
            CELLMETA_FILENAME = 'combined_metadata_2018May.txt'
            lincs = CombinedLINCS(dataset='combat', datadir=DATADIR, cellmeta_fname=CELLMETA_FILENAME)
        """
        if rna_norm == 'raw':
            DATASET = 'combined_rnaseq_data_lincs1000'
        elif rna_norm == 'source_scale':
            DATASET = 'combined_rnaseq_data_lincs1000_source_scale'
        elif rna_norm == 'combat':
            DATASET = 'combined_rnaseq_data_lincs1000_combat'
        else:
            raise ValueError(f'The passed dataset ({DATASET}) is not supported.')
            
        data_type = np.float32

        # Load RNA-Seq
        path = datadir / DATASET
        cols = pd.read_table(path, nrows=0, sep='\t')
        dtype_dict = {c: data_type for c in cols.columns[1:]}
        rna = pd.read_table(path, dtype=dtype_dict, sep='\t', na_values=na_values, warn_bad_lines=True)
        rna = self._keep_sources(rna, sources=sources) 

        # Load metadata
        meta = pd.read_table(datadir / cellmeta_fname, sep='\t')
        meta = self._update_metadata_comb_may2018(meta)
        
        # Merge rna and meta
        rna, meta = self._update_df_and_meta(rna, meta, on='Sample')

        if verbose:
            print(f'\nDataset: {DATASET}')
            print(f'rna {rna.shape}')
            if meta is not None:
                print(f'meta   {meta.shape}')
            print(rna['Sample'].map(lambda s: s.split('.')[0]).value_counts())
            
        self._rna, self._meta = rna, meta


    def _keep_sources(self, rna, sources=[]):
        """ Keep specific data sources.
		Args:
			sources (list) : list of strings indicating the sources/studies to extract.
                (e.g., source=['ccle', 'ctrp'])
		"""
        if len(sources) == 0:
            return rna

        if isinstance(sources, str):
            sources = [sources]
            
        if len(sources) > 0:
            sources = [s.lower() for s in sources]
            rna = rna.loc[rna['Sample'].map(lambda s: s.split('.')[0].lower() in sources), :].reset_index(drop=True)
        else:
            print('Empty list was passed to the arg `sources`. Returns the same dataframe.')

        return rna  


    def _update_metadata_comb_may2018(self, meta):
        """ Update the metadata of the combined RNA-Seq (Judith metadata):
        /nfs/nciftp/private/tmp/jcohn/metadataForRNASeq2018Apr/combined_metadata_2018May.txt
        Remove "unnecessary" columns.
        Use Argonne naming conventions (e.g. GDC rather than TCGA).
        """
        # Rename columns
        meta = meta.rename(columns={'sample_name': 'Sample',
                                    'dataset': 'source',
                                    'sample_category': 'category',
                                    'sample_descr': 'descr',
                                    'tumor_site_from_data_src': 'csite',
                                    'tumor_type_from_data_src': 'ctype',
                                    'simplified_tumor_site': 'simplified_csite',
                                    'simplified_tumor_type': 'simplified_ctype'
                                    })

        meta['source'] = meta['source'].map(lambda x: x.lower())
        meta['csite'] = meta['csite'].map(lambda x: x.strip())
        meta['ctype'] = meta['ctype'].map(lambda x: x.strip())
        meta['source'] = meta['source'].map(lambda x: 'gdc' if x=='tcga' else x)
        return meta

    
    def _update_df_and_meta(self, rna, meta, on='Sample'):
        """ Merge rna and meta on a column specified by `on`.
        Args:
            rna (df) : df rna
            meta (df) : df meta
        Returns:
            rna (df) : df rna updated
            meta (df) : df meta updated
        """
        rna = rna.copy()
        meta = meta.copy()
        df = pd.merge(meta, rna, how='inner', on=on).reset_index(drop=True)

        rna = df[['Sample'] + rna.columns[1:].tolist()]
        meta = df.drop(columns=rna.columns[1:].tolist())
        return rna, meta
    

    def get_rna(self):
        return self._rna.copy()
    
    
    def get_meta(self):
        return self._meta.copy()
    
    
    def get_subset(self, sources=[]):
        """ Get samples of the specified data sources (this is a getter method).
        Args:
            sources (list) : list of strings indicating the sources/studies to extract
        Returns:
            rna (df) : df rna for the data sources specified by `sources`
            meta (df) : df meta for the data sources specified by `sources`
        Example:
            cells_rna, cells_meta = lincs.get_subset(sources=['ccle','nci60'])
        """
        rna = self._rna.copy()
        meta = self._meta.copy()

        if len(sources) > 0:
            sources = [s.lower() for s in sources]
            rna = rna.loc[rna['Sample'].map(lambda s: s.split('.')[0].lower() in sources), :].reset_index(drop=True)
            rna, meta = self._update_df_and_meta(rna, meta, on='Sample')
        else:
            print('Empty list was passed to the arg `sources`. Returns the same dataframe.')

        return rna, meta
