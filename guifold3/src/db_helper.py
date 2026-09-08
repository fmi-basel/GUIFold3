# Copyright 2022 Friedrich Miescher Institute for Biomedical Research
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Georg Kempf, Friedrich Miescher Institute for Biomedical Research

from sqlalchemy import MetaData, Table, Column, String, Boolean, Integer, Float, DateTime, ForeignKey, MetaData, create_engine, func
from sqlalchemy.inspection import inspect
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import relationship, sessionmaker, scoped_session
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine, orm
import logging
from contextlib import contextmanager
import os
from shutil import copyfile
#Base = declarative_base()
logger = logging.getLogger('guifold3')

DB_REVISION = 13

def get_type(type):
    types = {'str': String,
             'int': Integer,
             'bool': Boolean,
             'float': Float}
    if not type is None:
        return types[type]
    else:
        return None


class DBMigration:
    def __init__(self):
        pass




class DBHelper:
    def __init__(self, shared_objects, db_path) -> None:
        self.shared_objects = shared_objects
        #self.add_attr(self.job)
        #self.add_attr(self.prj)
        self._DATABASE_NAME = 'guifold'
        self.db_path = db_path
        self.engine = create_engine('sqlite:///{}'.format(db_path), connect_args={'check_same_thread': False})
        name = '.'.join([__name__, self.__class__.__name__])
        self.logger = logging.getLogger(name)

    def init_db(self) -> None:
        self.reflect_tables()
        self.create_tables()
        self.conn = None
        session_factory = sessionmaker(bind=self.engine)
        self.Session = scoped_session(session_factory)
        self.sess = None


    def reflect_tables(self) -> None:
        """
        Create tables classes from shared object names and attributes.
        name = Name of the Table Object, e.g. Job, Project, etc.
        column = Name of the column
        :return:
        """
        tables = {}

        for obj in self.shared_objects:
            for var in vars(obj):
                logger.debug("Var: {}".format(var))
                column = getattr(obj, var)
                if hasattr(column, 'db'):
                    if not hasattr(column, 'db_relationship'):
                        db_relationship = None
                    else:
                        db_relationship = column.db_relationship
                    if not hasattr(column, 'db_foreign_key'):
                        foreign_key = None
                    else:
                        foreign_key = column.db_foreign_key
                    if not hasattr(column, 'db_backref'):
                        db_backref = None
                    else:
                        db_backref = column.db_backref
                    if column.db is True:
                        col = {'db_table': obj.db_table, 'name': var, 'type': get_type(column.type), 'primary_key': column.db_primary_key, 'db_relationship': db_relationship, 'db_backref': db_backref, 'foreign_key': foreign_key}
                        logger.debug(obj)
                        if not obj.db_table in tables:
                            tables[obj.db_table] = []
                        tables[obj.db_table].append(col)

        self.metadata = MetaData()
        self.metadata.reflect(bind=self.engine, autoload=True)
        #Column objects
        column_objs = {}
        #Relationship objects
        rel_objs = {}
        for name, columns in tables.items():
            for v in columns:
                logger.debug(v)
                if 'db_relationship' in col:
                    logger.debug(col['db_relationship'])
            #column_objs[name] = [relationship(col['db_relationship'], backref=col['db_table'], lazy='dynamic') if col['db_relationship'] is not None else Column(col['name'], col['type'], foreign_key=col['foreign_key'], primary_key=col['primary_key']) for col in columns]
            col_list = []
            rel_list = []
            for col in columns:
                if not col['db_relationship'] is None:
                    if not col['db_backref'] is None:
                        rel_list.append(
                            [col['name'], relationship(col['db_relationship'], passive_deletes=True, back_populates=col['db_backref'])])
                    else:
                        rel_list.append([col['name'], relationship(col['db_relationship'], passive_deletes=True)])
                elif not col['foreign_key'] is None:
                    col_list.append(Column(col['name'], col['type'], ForeignKey(col['foreign_key'], ondelete='CASCADE'), primary_key=col['primary_key']))
                else:
                    col_list.append(Column(col['name'], col['type'], primary_key=col['primary_key']))

            #column_objs[name] = [Column(col['name'], col['type'], ForeignKey(col['foreign_key']), primary_key=col['primary_key']) if not col['foreign_key'] is None else Column(col['name'], col['type'], primary_key=col['primary_key']) for col in columns]
            column_objs[name] = col_list
            logger.debug("col_list")
            logger.debug(col_list)
            rel_objs[name] = rel_list
            logger.debug("rel_list")
            logger.debug(rel_list)
        logger.debug("Content of column_objs dict")
        for k, v in column_objs.items():
            logger.debug(k)
            logger.debug(v)

        self.Base = automap_base()
        logger.debug("self dict")
        for name, objs in column_objs.items():
            logger.debug(self.__dict__)
            logger.debug(name.capitalize())
            self.__dict__[name.capitalize()] = type(name.capitalize(), (self.Base,),
                                                    {"__table__": Table(name, self.metadata, extend_existing=True, *objs)})
        for name, objs in rel_objs.items():
            for obj in objs:
                setattr(self.__dict__[name.capitalize()], obj[0], obj[1])
        logger.debug("adding relationships")
        self.Base.prepare(self.engine, reflect=True)
        logger.debug(self.__dict__)

    def create_tables(self):
        """
        Add the tables to the database if not present.
        :return:
        """
        self.metadata.create_all(self.engine, checkfirst=True)
        for _class in self.Base.classes:
            logger.debug(_class)

    def set_session(self, session):
        session = self.Session()
        self.sess = session

    @contextmanager
    def session_scope(self):
        """Provide a transactional scope around a series of operations."""
        session = self.Session()
        try:
            yield session
            session.commit()
            session.flush()
        except:
            session.rollback()
            raise
        finally:
            self.Session.remove()

    def backup_db(self, db_path):
        prev_db_revision = DB_REVISION - 1
        guifold_dir = os.path.join(os.path.expanduser("~"), f'.guifold')
        if not os.path.exists(guifold_dir):
            os.mkdir(guifold_dir)
        backup_db_path = os.path.join(guifold_dir, f'guifold.db.{prev_db_revision}')
        if not os.path.exists(backup_db_path):
            if os.path.exists(db_path):
                copyfile(db_path, backup_db_path)

    def update_job_type(self, sess):
        result_job = sess.query(self.Job).all()
        for job in result_job:
            if job.type is None or job.type == "":
                result_jobparams = sess.query(self.Jobparams).filter_by(job_id=job.id).one()
                #run_data_pipeline = bool(result_jobparams.run_data_pipeline)
                #run_inference = bool(result_jobparams.run_inference)
                #use_precomputed_msas = bool(result_jobparams.use_precomputed_msas)
                #batch_msas = bool(result_jobparams.batch_msas)
                pipeline = result_jobparams.pipeline
                if pipeline in ['run_data_pipeline', 'batch_msas']:
                    type = "features"
                elif pipeline in ['run_inference']:
                    type = "prediction"
                else:
                    type = "full"
                job.type = type
        sess.commit()

    def update_queue_jobid_regex(self, sess):
        try:
            result_settings = sess.query(self.Settings).filter_by(id=1).one()
            if not result_settings is None:
                if result_settings.queue_jobid_regex is None:
                    if result_settings.queue_submit == "sbatch":
                        logger.debug("Setting queue_jobid_regex")
                        result_settings.queue_jobid_regex = "\D*(\d+)\D*"
            sess.commit()
        except orm.exc.NoResultFound:
            logger.debug("No settings found. This is normal when the application is started for the first time.")
        except:
            logger.debug("Could not update jobidregex")


    def migrate_to_pipeline_cmb(self, sess):
        try:
            result_jobparams = sess.query(self.Jobparams).all()
            for row in result_jobparams:
                if row.pipeline is None:
                    if row.run_data_pipeline:
                        row.pipeline = 'run_data_pipeline'
                    elif row.batch_msas:
                        row.pipeline = 'batch_msas'
                    elif row.run_inference:
                        row.pipeline = 'run_inference'
                    elif not any([row.run_data_pipeline,
                                  row.batch_msas,
                                  row.use_precomputed_msas,
                                  row.run_inference]):
                        row.pipeline = 'full'
            sess.commit()
        except:
            logger.debug("Error while migrating to pipeline cmb.")

    def upgrade_db(self):
        self.logger.debug("Upgrading DB")
        self.backup_db(self.db_path)
        stmts = []
        with self.engine.connect() as conn:
            for stmt in stmts:
                try:
                    rs = conn.execute(stmt)
                except Exception as e:
                    logger.debug(e)
                    #traceback.print_exc()

