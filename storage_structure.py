from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.declarative import declarative_base
import pandas as pd
import numpy as np
import sqlalchemy
from sqlalchemy import create_engine, Table, Column, Float, DateTime, MetaData, String, TIMESTAMP, Integer, Boolean, PickleType, asc, desc, and_
from sqlalchemy import and_, or_
from datetime import datetime, timedelta
import math
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pathlib import Path

Base = declarative_base()

class Molecules(Base):
    __tablename__ = "Molecules"
    name = Column(String, primary_key=True)
    molecule = Column(PickleType)

class OpenEyeConformers(Base):
    __tablename__ = "OpenEyeConformers"

    id = Column(Integer, primary_key=True)
    conf_id = Column(Integer)
    name = Column(String)
    conformer = Column(PickleType)
    openeye_charges = Column(PickleType)
    ante_charges = Column(PickleType)
    oe_selected = Column(Boolean)
    rd_selected = Column(Boolean)

class RDKitConformers(Base):
    __tablename__ = "RDKitConformers"

    id = Column(Integer, primary_key=True)
    conf_id = Column(Integer)
    name = Column(String)
    conformer = Column(PickleType)
    openeye_charges = Column(PickleType)
    ante_charges = Column(PickleType)
    oe_selected = Column(Boolean)
    rd_selected = Column(Boolean)

def add_molecule(session, molecule):
    mol_name = molecule.name
    query = session.query(Molecules).filter(Molecules.name == mol_name).all()
    if len(query) > 0: # if a username already exists
        print("molecule name already exists")
        return
    
    # try inserting the molecule
    try:
        session.add(Molecules(name=mol_name, molecule=molecule))
    except Exception as e:
        print(e)
        return
    # try commiting 
    try:
        session.commit()
    except Exception as e:
        print(e)
        return
    return

def add_conformer_charges(session, table, name, conf_id, conformer, openeye_charges, ante_charges, oe_selected, rd_selected):
    query = session.query(table).filter(and_(table.name == name, table.conf_id == conf_id)).all()
    if len(query) > 0: # if a username already exists
        print("molecule name already exists")
        return
    # try inserting info
    try:
        session.add(table(name=name, 
                          conf_id=conf_id,
                          conformer = conformer,
                          openeye_charges = openeye_charges,
                          ante_charges = ante_charges,
                          oe_selected = oe_selected,
                          rd_selected = rd_selected))
    except Exception as e:
        print(e)
        return

    # try commiting
    try:
        session.commit()
    except Exception as e:
        print(e)
        return
        
    return

def already_exists(session, name):
    query = session.query(Molecules).filter(Molecules.name == name).all()
    if len(query) > 0: # if a username already exists
        return True
    else:
        return False