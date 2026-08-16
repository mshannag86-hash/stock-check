-- Schema fuer Aktien-Analyse-Assistent: Nutzer-Accounts + Watchlist.
-- Ausfuehren in Supabase -> SQL Editor -> New query.
--
-- RLS aktiviert, aber ohne Policies (default deny) -- die App greift
-- ausschliesslich ueber den service_role Key zu (bypasst RLS by design,
-- bleibt server-seitig in den Streamlit-Secrets, nie im Browser sichtbar).
-- Falls der falsche (anon) Key jemals versehentlich landet, kommt niemand
-- an die Daten, weil keine Policy existiert.

create extension if not exists pgcrypto;

create table if not exists users (
    id uuid primary key default gen_random_uuid(),
    username text unique not null,
    password_hash text not null,
    created_at timestamptz default now()
);

create table if not exists watchlist (
    id uuid primary key default gen_random_uuid(),
    user_id uuid not null references users(id) on delete cascade,
    ticker text not null,
    added_at timestamptz default now(),
    unique (user_id, ticker)
);

alter table users enable row level security;
alter table watchlist enable row level security;
