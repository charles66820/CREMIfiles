all:
	ocamlc -c images.mli
	ocamlc -c images.ml
	ocamlc -c zones.ml
run: all
	ocaml visu-zones.ml
build:
	#ocamlbuild -use-ocamlfind main.d.byte

clean:
	rm -f images.cmi images.cmo zones.cmi zones.cmo

	#ocamlbuild -clean
.PHONY: build clean all run

