project_name = anbn

srcdir = src
bindir = bin

bison = bison
bison_options = --report=all --report itemsets -Wcex

parserjava = $(srcdir)/Parser.java

srcfiles = $(wildcard $(srcdir)/*.java) $(parserjava)

run: bin
	java -cp $(bindir) Parser

test: input.txt bin
	java -cp $(bindir) Parser $<

bin: parser $(bindir)/Parser.class

$(bindir)/Parser.class: $(srcfiles)
	mkdir -p $(bindir)
	javac -sourcepath $(srcdir) -d $(bindir) $(srcdir)/Parser.java

parser: $(srcdir)/CustomLexer.java $(srcdir)/Parser.java

$(srcdir)/CustomLexer.java: lexer/CustomLexer.jflex
	java -jar lib/jflex-full-1.8.2.jar $< -d $(srcdir)

$(srcdir)/Parser.java: parser/Parser.y
	$(bison) $(bison_options) $< -o $@

clean:
	rm $(bindir)/* || true
	rm $(srcdir)/Parser.java || true
	rm $(srcdir)/CustomLexer.java || true
