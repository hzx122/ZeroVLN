"""habitat_sim-backed asset sets that are not the R2R/MP3D VLN dataset.

``hsim.env.HabSceneEnv`` is the shared simulator; ``hsim.datasets`` describes the
individual asset sets (PARTNR / HSSD, Habitat-HAB / ReplicaCAD).  Import lazily:
pulling this package in loads habitat_sim, which needs ``libOpenGL.so.0`` on
``LD_LIBRARY_PATH`` before the interpreter starts (see ``src/run_console.sh``).
"""
