-- Create role if not exists
DO $$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'metadata_user') THEN
      CREATE ROLE metadata_user LOGIN PASSWORD '123';
   END IF;
END
$$;

-- Create database if not exists
DO $$
BEGIN
   IF NOT EXISTS (SELECT FROM pg_database WHERE datname = 'metadata_db') THEN
      CREATE DATABASE metadata_db OWNER metadata_user;
   END IF;
END
$$;

\connect metadata_db

-- Schema privileges
GRANT ALL ON SCHEMA public TO metadata_user;