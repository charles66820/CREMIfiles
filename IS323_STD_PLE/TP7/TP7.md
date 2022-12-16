# TP4

## Notes

`HBase` work with a namespace system. Example with table access :

`<namespace>:<tableName>`
``

Command to access to `HBase` shell :

```bash
hbase shell
```

## Commands

Get current user info :

```HBase
whoami
```

Create a table :

> `create "<table name>","<column family>"`

```HBase
t = create "cgoedefroit:cgoedefroit",  "loc", "name", "reg", "measure"
```

The command `list` to get the table list.
The command `describe "cgoedefroit:cgoedefroit"` to get table information. (e.g. `t.describe`)

To remove a table :

```HBase
disable "cgoedefroit:<tableName>"
drop "cgoedefroit:<tableName>"
```

To get table as variable :

```HBase
t = get_table "cgoedefroit:cgoedefroit"
```

To alter an table :

> we indicate we want to keep five versions of the data

```HBase
t.disable
alter "cgoedefroit:cgoedefroit", {NAME => "loc", VERSIONS => 5}
t.enable
```

> `put "<tableName>", "<rowName>", "<colName>", "<value>"`

<!-- , ts1, {ATTRIBUTES=>{'mykey'=>'myvalue'}} -->

```HBase
t.put 0, "loc:x", -20.3333333
t.put 0, "loc:y", 30.0333333
t.put 0, "name:city", "zvishavane"
t.put 0, "name:accentCity", "Zvishavane"
t.put 0, "reg:country", "zw"
t.put 0, "reg:code", "07"
t.put 0, "measure:pop", 79876
```

The command `scan "cgoedefroit:cgoedefroit"` to get table data. (e.g. `t.scan`)

The command `get` permit to obtain an specific row with a key. (e.g `t.get 0`)

> example with args

```HBase
t.put 0, "loc:x", -20.0
t.put 0, "measure:pop", 500
get "cgoedefroit:cgoedefroit", 0, {COLUMN => ["loc", "measure"], VERSIONS => 5 }
```

The command `delete` remove an specific data at a precise timestamp.

```HBase
t.delete 0, "loc:x",  1671189236795
```

```HBase
t.put 1, "loc:x", 42.4833333
t.put 1, "loc:y", 1.4666667
t.put 1, "name:city", "aixas"
t.put 1, "name:accentCity", "Aix�s"
t.put 1, "reg:country", "ad"
t.put 1, "reg:code", "06"
t.put 1, "measure:pop", 0
```

```HBase
scan "cgoedefroit:cgodefroit", {FILTER=>"SingleColumnValueFilter("reg","code",=,'binary:"06")"}
```
