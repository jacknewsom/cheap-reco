#!/usr/bin/env python

import os
import ROOT
from optparse import OptionParser
import subprocess
from array import array

t_x = array( 'd', [0.] )
t_y = array( 'd', [0.] )
t_z = array( 'd', [0.] )

offset = ROOT.TVector3( 0., 73.94, 479.25 )
dimensions = ROOT.TVector3( 713.1, 300., 507.3 ) # cm
padpitch = 0.3 # cm
timeres = 0.3 # cm, resolution is better in x dimension because timing

def getVoxyl( pos ):
    x = int( (pos.x() - dimensions.x()/2.) / timeres )
    y = int( (pos.y() - dimensions.y()/2.) / padpitch )
    z = int( (pos.z() / padpitch )
    return int(x), int(y), int(z)

def loop( events, tgeo, hdf5file ):

    event = ROOT.TG4Event()
    events.SetBranchAddress("Event",ROOT.AddressOf(event))

    N = events.GetEntries()
    for ient in range(0, N):
        events.GetEntry(ient) # Load edep-sim event

        info = ROOT.ProcInfo_t()
        ROOT.gSystem.GetProcInfo(info)
        vm = info.fMemVirtual
        rm = info.fMemResident

        if ient % 100 == 0:
            print "Event %d of %d, VM = %1.2f MB, RM = %1.2f MB" % (ient,N,vm/1000.,rm/1000.)

        # There is only ever one primary vertex per event, but in principle there could be more, so we have to do this
        for ivtx,vertex in enumerate(event.Primaries):

            # Save the vertex position so we can determine if it's in the fiducial volume or not
            vtx = vertex.Position.Vect()

            # We probably want to save information about the true neutrino interaction, so that we can 
            # see how the reco does for different types of interactions
            n_FinalStateParticles = 0
            for ipart,particle in enumerate(vertex.Particles):
                mom = particle.Momentum
                pdg = particle.PDGCode
                ke = mom.E() - mom.M()
                if pdg < 9999: 
                    n_FinalStateParticles += 1 # don't count nuclear fragments, bindinos, etc.

            ArCube_hits = []
            for det in event.SegmentDetectors:
                if det.first == "ArgonCube":
                    ArCube_hits += det.second

            voxyl_energy = {}
            for k, edep in enumerate(ArCube_hits):
                if edep.EnergyDeposit < 0.01: # Some energy threshold
                    continue

                node = tgeo.FindNode( edep.Start.X(), edep.Start.Y(), edep.Start.Z())
                if "volLArActive" not in node.GetName():
                    continue

                hPos = ROOT.TVector3( edep.Start.X()/10., edep.Start.Y()/10., edep.Start.Z()/10. )
                hPos += offset # (0,0,0) is in the middle of the upstream face of the active volume

                #hPos.x(), hPos.y(), hPos.z() are the coordinates
                x,y,z = getVoxyl( hPos ) # x,y,z are integer coordinates of voxyls
                if (x,y,z) not in voxyl_energy:
                    voxyl_energy[(x,y,z)] = edep.EnergyDeposit
                else:
                    voxyl_energy[(x,y,z)] += edep.EnergyDeposit

if __name__ == "__main__":

    ROOT.gROOT.SetBatch(1)

    parser = OptionParser()
    parser.add_option('--outfile', help='Output file name', default="out.root")
    parser.add_option('--topdir', help='Input file top directory', default="/pnfs/dune/persistent/users/marshalc/neutronSim")
    parser.add_option('--first_run', type=int, help='First run number', default=1001)
    parser.add_option('--last_run', type=int, help='Last run number', default=1001)
    parser.add_option('--rhc', action='store_true', help='Reverse horn current', default=False)
    parser.add_option('--geom',help='top volume of interactions', default="DetEnclosure")
    parser.add_option('--grid',action='store_true', help='Grid mode')

    (args, dummy) = parser.parse_args()

    # Get the number of protons on target from the underlying GENIE files -- this is to determine how many events per spill
    # This is a hack, you should not do stuff like this, avert your eyes
    rhcarg = "--rhc" if args.rhc else ""
    gridarg = "--grid" if args.grid else ""
    cppopts = ['./getPOT', '--topdir', args.topdir, '--first', str(args.first_run), '--last', str(args.last_run), '--geom', args.geom, rhcarg, gridarg]
    sp = subprocess.Popen(cppopts, stdout=subprocess.PIPE, stderr=None)
    the_POT = float(sp.communicate()[0])
   
    tgeo = None

    neutrino = "neutrino" if not args.rhc else "antineutrino"
    horn = "FHC" if not args.rhc else "RHC"

    # Create HDF5 output file here probably
    hdf5file = None

    for run in range( args.first_run, args.last_run+1 ):
        fname = None
        if args.grid:
            fname = "%s.%d.edepsim.root" % (neutrino, run)
        else:
            fname = "%s/EDep/%s/%s/%s.%d.edepsim.root" % (args.topdir, horn, args.geom, neutrino, run)
        if not os.access( fname, os.R_OK ):
            print "Can't access file: %s" % fname
            continue
        tf = ROOT.TFile( fname )
        events = tf.Get( "EDepSimEvents" )

        if tgeo is None:
            tf.MakeProject("EDepSimEvents","*","RECREATE++")
            tgeo = tf.Get( "EDepSimGeometry" )

        print "Looping over: %s" % fname
        # Loop over one edep-sim input file
        loop( events, tgeo, hdf5file )
        tf.Close()


