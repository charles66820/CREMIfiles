let est_une_figure carte = match carte with As(couleur) -> true | Roi(couleur) -> true | Dame(couleur) -> true | Valet(couleur) -> true | _ -> false;;
