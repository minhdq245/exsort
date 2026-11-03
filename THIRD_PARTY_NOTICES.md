# Third-party components

- `exsort/vendor/ocsort/{association.py,ocsort.py,kalmanfilter.py}`: retained byte-for-byte from the local OC-SORT dependency used in the experiments. Upstream: https://github.com/noahcao/OC_SORT ; local checkout revision `8462e7e729a93ccd3bd995c0a79a890336cb3a0b`. MIT notice retained in `licenses/OC-SORT-MIT.txt`. `ocsort.py` credits Alex Bewley's SORT. `kalmanfilter.py` retains Roger R. Labbe Jr.'s FilterPy copyright and MIT notice; see `licenses/FilterPy-LICENSE`.
- The coordinate helpers and exsort implementations descend from the HALO-SORT research workspace. `history/source_provenance.json` records original source SHA256 values. The portable release changes import paths; unselected experimental branches remain visible for ablation.
- FilterPy is an installed dependency under its upstream MIT license: https://github.com/rlabbe/filterpy . Its license copy accompanies the vendored derivative filter.
- TrackEval is installed from its official repository at a pinned commit, not vendored: https://github.com/JonathonLuiten/TrackEval . Keep its upstream license when redistributing it.
- SMAC3/ConfigSpace are installed dependencies with their own upstream licenses. Consult https://github.com/automl/SMAC3 and https://github.com/automl/ConfigSpace .
- YOLOX/TrackTrack are external optional dependencies, not vendored or relicensed here. Follow their original licenses: https://github.com/kamkyu94/TrackTrack and https://github.com/ifzhang/ByteTrack . No TrackTrack ReID code, model weights or dataset images are included.

The root license applies to the project's own contributions. It does not replace the notices or terms of third-party components, datasets or weights.
