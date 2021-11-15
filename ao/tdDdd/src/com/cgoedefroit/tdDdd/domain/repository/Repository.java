package com.cgoedefroit.tdDdd.domain.repository;

import com.cgoedefroit.tdDdd.domain.aggregation.Aggregation;

public interface Repository {
    public Aggregation findById(int id);
    public void save (Aggregation obj);
    public void update(Aggregation obj);
}