#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
@File   :   ms.py
@Author :   Song
@Time   :   2020/1/20 9:41
@Contact:   songjian@westlake.edu.cn
@intro  :
'''
import time
import numpy as np
import multiprocessing as mp
from decimal import Decimal
import pyteomics.mzxml
import pyteomics.mzml
import pyteomics.mass
import utils_tri

try:
    profile
except NameError:
    profile = lambda x: x


class mz_Reader():

    def __init__(self, fpath, acquisition_type):
        self.mz = None
        self.all_rt = np.array([])
        self.TimeUnit = None
        self.SwathSettings = np.array([])

        # type
        fpath = str(fpath)
        self.suffix = fpath.split('.')[-1].lower()
        self.acquisition_type = acquisition_type

        self.load(fpath)
        self.init()

    def size(self):
        return len(self.mz)

    def load(self, file_path):
        if self.suffix == 'mzxml':
            self.mz = pyteomics.mzxml.MzXML(file_path, use_index=True)
        elif self.suffix == 'mzml':
            self.mz = pyteomics.mzml.MzML(file_path, use_index=True)

    def init(self):
        self.get_time_unit()
        self.__load_to_memory()
        if self.acquisition_type == 'DIA':
            self.__init_swath_window_array()
            self.check()

    def check(self):
        assert len(self.all_levels) % len(self.SwathSettings) == 0, 'Swath scan nums != K*(MS1+N-MS2)'

    def get_time_unit(self):
        if self.suffix == 'mzxml':
            if self.mz[self.size() - 1]['retentionTime'] < 3 * 60:
                self.TimeUnit = 'minute'
            else:
                self.TimeUnit = 'second'
        elif self.suffix == 'mzml':
            if self.mz[self.size() - 1]['scanList']['scan'][0]['scan start time'] < 5 * 60:
                self.TimeUnit = 'minute'
            else:
                self.TimeUnit = 'second'

    def process_worker(self, idx_start, idx_end):
        rts, levels, peaks_mz, peaks_intensity = [], [], [], []

        for idx in range(idx_start, idx_end):
            scan = self.mz[idx]
            if self.suffix == 'mzxml':
                rts.append(np.float32(scan['retentionTime']))
                levels.append(np.int8(scan['msLevel']))
            elif self.suffix == 'mzml':
                rts.append(np.float32(scan['scanList']['scan'][0]['scan start time']))
                levels.append(np.int8(scan['ms level']))
            peaks_mz.append(scan['m/z array'].astype(np.float32))
            peaks_intensity.append(scan['intensity array'].astype(np.float32))

        return {'rt': rts, 'level': levels, 'mz': peaks_mz, 'intensity': peaks_intensity}

    def __load_to_memory(self):
        self.all_mz, self.all_intensity, self.all_levels, self.all_rt = [], [], [], []

        # multiprocess load data
        cpu_num = mp.cpu_count()
        process_num = int(cpu_num / 2)  # default cores / 2
        process_num = 8 if process_num >= 8 else process_num
        pool = mp.Pool(process_num)
        slices = np.ceil(np.linspace(0, len(self.mz), process_num + 1)).astype(int)

        results = [pool.apply_async(self.process_worker, args=(slices[i], slices[i + 1])) for i in range(process_num)]
        results = [r.get() for r in results]  # get dict
        pool.close()
        pool.join()

        for result in results:
            self.all_rt.extend(result['rt'])
            self.all_levels.extend(result['level'])
            self.all_mz.extend(result['mz'])
            self.all_intensity.extend(result['intensity'])

        self.all_rt = np.array(self.all_rt)
        self.all_levels = np.array(self.all_levels)
        self.all_mz = np.array(self.all_mz)
        self.all_intensity = np.array(self.all_intensity)

        if self.acquisition_type == 'DIA':
            # MS2 win numbers：
            self.raw_ms1_idx = np.where(self.all_levels == 1)[0]
            cycles_ms2_num = np.diff(self.raw_ms1_idx) - 1
            self.windows_num = np.bincount(cycles_ms2_num).argmax()

            bad_slice = []

            # boundary of start
            if self.raw_ms1_idx[0] != 0:
                bad_slice.extend(range(0, self.raw_ms1_idx[0]))
            # boundary of end
            if len(self.mz) - self.raw_ms1_idx[-1] != self.windows_num + 1:
                bad_slice.extend(range(self.raw_ms1_idx[-1], len(self.mz)))
            # others
            for cycle_idx in np.where(cycles_ms2_num != self.windows_num)[0]:
                bad_slice.extend(range(self.raw_ms1_idx[cycle_idx], self.raw_ms1_idx[cycle_idx + 1]))

            if len(bad_slice) > 0:
                good_slice = np.arange(len(self.all_levels))
                good_slice = good_slice[~np.isin(good_slice, bad_slice)]

                self.all_rt = self.all_rt[good_slice]
                self.all_levels = self.all_levels[good_slice]
                self.all_mz = self.all_mz[good_slice]
                self.all_intensity = self.all_intensity[good_slice]

            # empty scan
            all_scans_len = np.array(list(map(len, self.all_mz)), dtype=np.int32)
            zero_scan_num = (all_scans_len == 0).sum()
            for zero_idx in np.where(all_scans_len == 0)[0]:
                self.all_mz[zero_idx] = np.array([888.], dtype=np.float32)
                self.all_intensity[zero_idx] = np.array([0.], dtype=np.float32)

        self.all_ms1_idx = np.where(self.all_levels == 1)[0]

        # time
        if self.TimeUnit == 'minute':
            self.all_rt = self.all_rt * 60.

    def get_ms1_all_rt(self):
        num_windows = len(self.SwathSettings) - 1
        scans_ms1_rt = self.all_rt[::(num_windows + 1)]
        return scans_ms1_rt

    def get_scan_mz(self, idx):
        return self.all_mz[idx]

    def get_scan_intensity(self, idx):
        return self.all_intensity[idx]

    def get_scan_peaks(self, idx):
        '''Peaks: [mz, intensity]'''
        mz = self.get_scan_mz(idx)
        inten = self.get_scan_intensity(idx)
        return (mz, inten)

    def get_current_scan_window(self, idx):
        idx = int(idx)

        if self.suffix == 'mzxml':
            middle = Decimal(str(self.mz[idx]['precursorMz'][0]['precursorMz']))
            width = Decimal(str(self.mz[idx]['precursorMz'][0]['windowWideness']))
            return (float(middle - width / 2), float(middle + width / 2))

        elif self.suffix == 'mzml':
            middle = Decimal(
                str(self.mz[idx]['precursorList']['precursor'][0]['isolationWindow']['isolation window target m/z']))
            lower_offset = Decimal(str(
                self.mz[idx]['precursorList']['precursor'][0]['isolationWindow']['isolation window lower offset']))
            upper_offset = Decimal(str(
                self.mz[idx]['precursorList']['precursor'][0]['isolationWindow']['isolation window upper offset']))
            return (float(middle - lower_offset), float(middle + upper_offset))

    def __init_swath_window_array(self):
        if len(self.SwathSettings) == 0:
            swath = []
            # scan idx
            while True:
                idx = np.random.choice(len(self.raw_ms1_idx) - 2)
                if self.raw_ms1_idx[idx + 1] - self.raw_ms1_idx[idx] == self.windows_num + 1:
                    break

            idx_start = self.raw_ms1_idx[idx] + 1  # ms2
            idx_end = self.raw_ms1_idx[idx + 1]

            # cycle
            while idx_start < idx_end:
                swath_windom = self.get_current_scan_window(idx_start)
                if swath_windom not in swath:
                    swath.append(swath_windom)
                    idx_start += 1
            self.swath_pair = swath
            swath = np.array([_ for item in swath for _ in item])

            # overlap
            result = []
            if np.min(np.diff(swath)) < 0:  # overlap
                result.append(swath[0])
                idx = 1
                while idx + 1 < len(swath) - 1:
                    result.append(np.mean((swath[idx], swath[idx + 1])))
                    idx += 2
                result.append(swath[-1])
                self.SwathSettings = np.array(result)
            elif np.min(np.diff(swath)) == 0:
                self.SwathSettings = np.sort(np.unique(swath))

    @profile
    def get_ms2_xics_by_fg_mz(self, idx_start, idx_end, ms2_win_idx, mz_query, ppm_tolerance):
        num_windows = len(self.SwathSettings) - 1

        # cycle
        result_xics, result_rts = [], []
        for cycle_idx, scan_idx in enumerate(range(idx_start, idx_end + 1, num_windows + 1)):
            result_rts.append(self.all_rt[scan_idx])
            peaks_mz = self.all_mz[scan_idx + ms2_win_idx]
            peaks_int = self.all_intensity[scan_idx + ms2_win_idx]
            xic_v_6 = utils_tri.find_ok_matches(peaks_mz, peaks_int, mz_query.astype(np.float32), ppm_tolerance)

            result_xics.append(xic_v_6)

        result_xics = np.array(result_xics).T
        result_rts = np.array(result_rts)

        return result_xics, result_rts


    @profile
    def get_ms1_ms2_xics_by_lib_mz(self, idx_start, idx_end, ms2_win_idx, query_mz, ppm_ms1, ppm_ms2):
        query_pr_mz = query_mz[0:3]
        query_fg_mz = query_mz[3:]

        num_windows = len(self.SwathSettings) - 1

        result_ms1_xics, result_ms1_rts = [], []
        result_ms2_xics, result_ms2_rts = [], []

        for cycle_idx, scan_idx in enumerate(range(idx_start, idx_end + 1, num_windows + 1)):
            # MS1
            result_ms1_rts.append(self.all_rt[scan_idx])
            peaks_mz = self.all_mz[scan_idx]
            peaks_int = self.all_intensity[scan_idx]
            xic_v_3 = utils_tri.find_ok_matches(peaks_mz, peaks_int, query_pr_mz.astype(np.float32), ppm_ms1)
            result_ms1_xics.append(xic_v_3)

            # MS2
            result_ms2_rts.append(self.all_rt[scan_idx + ms2_win_idx])
            peaks_mz = self.all_mz[scan_idx + ms2_win_idx]
            peaks_int = self.all_intensity[scan_idx + ms2_win_idx]
            xic_v_6 = utils_tri.find_ok_matches(peaks_mz, peaks_int, query_fg_mz.astype(np.float32), ppm_ms2)
            result_ms2_xics.append(xic_v_6)

        result_ms1_xics = np.array(result_ms1_xics).T
        result_ms1_rts = np.array(result_ms1_rts)

        result_ms2_xics = np.array(result_ms2_xics).T
        result_ms2_rts = np.array(result_ms2_rts)

        return result_ms1_xics, result_ms1_rts, result_ms2_xics, result_ms2_rts

def load_ms(ms_file, type):
    start_time = time.time()
    ms = mz_Reader(ms_file, type)

    print('{:<30}{:.2f}s'.format('ms loading time', time.time() - start_time))

    return ms
