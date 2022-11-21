# the selected molecules for all tests completed in this folder
from functools import partial
import sys, getopt, os
from pathlib import Path
from tempfile import TemporaryDirectory
from openeye.oechem import All
from rdkit.Chem import AllChem
from openff.toolkit.topology.molecule import FrozenMolecule, Molecule
from openff.toolkit.utils.toolkits import RDKitToolkitWrapper, OpenEyeToolkitWrapper
from openff.toolkit.utils import UndefinedStereochemistryError
from simtk import unit
import pandas as pd
import numpy as np
import subprocess
from distutils.spawn import find_executable
from openff.toolkit.utils.utils import temporary_cd 
from copy import deepcopy
from storage_structure import *
from MolDatabase import MolDatabase, confs_same
import time

def create_database(name):
    engine = create_engine(name)
    Base.metadata.create_all(engine)
    engine.dispose()
    
def populate_database(name):
    engine = create_engine(name)
    Session = sessionmaker(bind=engine) # define a session to work on the engine we just made
    session = Session()

    try:
        n_confs = 300
        for mol, status in MolDatabase():
            print(mol.name)
            try:
                if already_exists(session, mol.name):
                    print(f"molecule already processed under the name: {mol.name}")
                    continue
                add_molecule(session, mol)
                # RDKit generated conformers
                time1 = time.time()
                mols_generated = MolDatabase([]).generate_confs(mol, n_conformers=n_confs, method='rdkit')
                time2 = time.time()
                time_to_generate_confs = time2 - time1
                if mols_generated:
                    original_mol = deepcopy(mol)
                    for total_conformers_used in range(1, n_confs):
                        mol_copy1 = Molecule(mol)
                        mol_copy2 = Molecule(mol)
                        # put in the first n conformers
                        conformers = original_mol.conformers[0:total_conformers_used]
                        mol_copy1._conformers = conformers
                        mol_copy2._conformers = conformers
                        time3 = time.time()
                        RDKitToolkitWrapper().apply_elf_conformer_selection(mol_copy1, limit=1)
                        time4 = time.time()
                        rd_selected_conf = mol_copy1.conformers[0]
                        time5 = time.time()
                        OpenEyeToolkitWrapper().apply_elf_conformer_selection(mol_copy2, limit=1)
                        time6 = time.time()
                        oe_selected_conf = mol_copy2.conformers[0]
                        time_to_apply_rd_elf = time4 - time3
                        time_to_apply_oe_elf = time6 - time5

                        if confs_same(conformers[-1], rd_selected_conf):
                            rd_selected = True
                        else:
                            rd_selected = False

                        if confs_same(conformers[-1], oe_selected_conf):
                            oe_selected = True
                        else:
                            oe_selected = False

                        # add to the database
                        add_conformer_charges(session, 
                                            RDKitConformers,
                                            mol.name,
                                            total_conformers_used,
                                            time_to_generate_confs,
                                            time_to_apply_oe_elf,
                                            time_to_apply_rd_elf,
                                            oe_selected,
                                            rd_selected)
                else:
                    print("some problem with conformer generation")
                # Openeye generated conformers
                time1 = time.time()
                mols_generated = MolDatabase([]).generate_confs(mol, n_conformers=n_confs, method='openeye')
                time2 = time.time()
                time_to_generate_confs = time2 - time1
                if mols_generated:
                    original_mol = deepcopy(mol)
                    for total_conformers_used in range(1, n_confs):
                        mol_copy1 = Molecule(mol)
                        mol_copy2 = Molecule(mol)
                        # put in the first n conformers
                        conformers = original_mol.conformers[0:total_conformers_used]
                        mol_copy1._conformers = conformers
                        mol_copy2._conformers = conformers
                        time3 = time.time()
                        RDKitToolkitWrapper().apply_elf_conformer_selection(mol_copy1, limit=1)
                        time4 = time.time()
                        rd_selected_conf = mol_copy1.conformers[0]
                        time5 = time.time()
                        OpenEyeToolkitWrapper().apply_elf_conformer_selection(mol_copy2, limit=1)
                        time6 = time.time()
                        oe_selected_conf = mol_copy2.conformers[0]
                        time_to_apply_rd_elf = time4 - time3
                        time_to_apply_oe_elf = time6 - time5

                        if confs_same(conformers[-1], rd_selected_conf):
                            rd_selected = True
                        else:
                            rd_selected = False

                        if confs_same(conformers[-1], oe_selected_conf):
                            oe_selected = True
                        else:
                            oe_selected = False

                        # add to the database
                        add_conformer_charges(session, 
                                            OpenEyeConformers,
                                            mol.name,
                                            total_conformers_used,
                                            time_to_generate_confs,
                                            time_to_apply_oe_elf,
                                            time_to_apply_rd_elf,
                                            oe_selected,
                                            rd_selected)
                    
                else:
                    print("some problem with conformer generation")
            except Exception as e:
                print(e)
                print("some problem with charge or conformer assignment")

    except Exception:
        pass
    finally:
        print("closing and disposing of database") # always do this, especially if you want to run this script continuously
        session.close()
        engine.dispose()


if __name__ == "__main__":
    name = "sqlite:///ELF_selection_tests/conformer_elf_selection_database.db"
    create_database(name)
    populate_database(name)
